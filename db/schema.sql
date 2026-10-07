-- RAG 知识库表结构（MySQL 8 / 5.7，utf8mb4）
--
-- 为什么这个文件必须存在：db/parent_store.py 与 db/document_store.py 依赖
-- parents / documents 两张表，但此前仓库里没有任何 DDL —— 换个环境 clone 下来
-- 根本建不出库，且"缺表"不会报错，只会被 db.mysql 收敛成 MySQLUnavailable，
-- 然后被 _save_parents / _record_document 静默吞掉，检索悄悄降级成纯子块。
--
-- 用法：python -m db.init_schema   （幂等，可重复执行）
--
-- 字符集必须是 utf8mb4：父块正文含大量 OCR 文本与生僻字，
-- MySQL 的 3 字节 "utf8" 别名存不下，会直接报 Incorrect string value。
-- 刻意不写 COLLATE：让不同 MySQL 版本各自用默认排序规则（5.7 上没有 0900_ai_ci）。

CREATE TABLE IF NOT EXISTS `parents` (
  `parent_id`   varchar(32)  NOT NULL COMMENT '父块主键 = "p_" + sha1(source|order_idx) 前 30 位',
  `doc_id`      varchar(32)  NOT NULL COMMENT '所属文档 = "d_" + sha1(source) 前 30 位',
  `source`      varchar(500) NOT NULL COMMENT '原始文件绝对路径（文档身份就是它，改路径等于换文档）',
  `source_hash` char(40)     NOT NULL COMMENT 'sha1(source)，用于按文件整体替换父块',
  `breadcrumb`  varchar(500) DEFAULT NULL COMMENT '章节路径，如 "手册.pdf > 第2章 > 2.3"（引用可核验用）',
  `page_start`  int          DEFAULT NULL,
  `page_end`    int          DEFAULT NULL,
  `order_idx`   int          DEFAULT NULL COMMENT '父块在文档内的序号',
  `char_len`    int          DEFAULT NULL COMMENT '正文的字符数（Python len，不是字节数）',
  `text`        mediumtext   NOT NULL COMMENT '父块全文：送给 LLM 读的大块',
  PRIMARY KEY (`parent_id`),
  KEY `idx_source_hash` (`source_hash`),
  KEY `idx_doc_id` (`doc_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='父块存储：只按 id 取全文，不参与向量/BM25 检索';

CREATE TABLE IF NOT EXISTS `documents` (
  `doc_id`           varchar(32)  NOT NULL COMMENT '文档主键 = "d_" + sha1(source) 前 30 位',
  `filename`         varchar(255) NOT NULL COMMENT '文件名（不含目录）',
  `source`           varchar(500) NOT NULL COMMENT '原始文件绝对路径',
  `uploaded_at`      datetime     NOT NULL COMMENT '最近一次入库时间',
  `parsed_status`    varchar(20)  NOT NULL COMMENT 'ok / failed',
  `error_msg`        text                  COMMENT '解析或入库失败原因（failed 时写）',
  `chunk_count`      int          DEFAULT 0 COMMENT '子块数（Chroma）',
  `parent_count`     int          DEFAULT 0 COMMENT '父块数（MySQL）',
  `chunk_schema_ver` varchar(20)  DEFAULT NULL COMMENT '切分参数版本，见 rag/structure.py CHUNK_SCHEMA_VER',
  `tenant_id`        varchar(32)  NOT NULL DEFAULT 'default' COMMENT '租户：数据隔离维度',
  `owner_id`         varchar(64)  DEFAULT NULL COMMENT '上传者；visibility=private 时只有他能看',
  `visibility`       varchar(16)  NOT NULL DEFAULT 'tenant' COMMENT 'tenant / private / public',
  `status`           varchar(16)  NOT NULL DEFAULT 'draft' COMMENT 'draft 待审核 / published 已发布 / archived 下架',
  `published_at`     datetime     DEFAULT NULL COMMENT '审核通过时间',
  `published_by`     varchar(64)  DEFAULT NULL COMMENT '审核人 user_id',
  PRIMARY KEY (`doc_id`),
  KEY `idx_uploaded` (`uploaded_at`),
  KEY `idx_acl` (`tenant_id`, `status`, `visibility`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='文档元数据：重建一致性校验 + 切分参数版本 + 访问控制与审核状态';

-- ════════════════════════════════════════════════════════════════════
-- 认证与授权
--
-- 三条设计约定（都是"失败关闭"方向）：
--   1. 密码只存 bcrypt 哈希，永远不存明文、也不写进日志；
--   2. 刷新令牌只存 sha256 哈希 —— 库被读走也不能直接拿来登录；
--   3. 审计日志单独一张表：企业要能回答"谁在什么时候做了什么"。
-- ════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS `users` (
  `user_id`       varchar(32)  NOT NULL COMMENT '主键 = "u_" + 26 位随机串',
  `username`      varchar(64)  NOT NULL COMMENT '登录名（同一租户内唯一）',
  `display_name`  varchar(64)  DEFAULT NULL COMMENT '展示名',
  `password_hash` varchar(200) NOT NULL COMMENT 'bcrypt 哈希（自带算法与代价参数）',
  `role`          varchar(20)  NOT NULL COMMENT 'admin / kb_admin / operator / user',
  `customer_id`   varchar(32)  DEFAULT NULL COMMENT '业务客户号：认证层据此把账号映射到订单/工单等业务数据',
  `tenant_id`     varchar(32)  NOT NULL DEFAULT 'default' COMMENT '租户：数据隔离维度',
  `status`        varchar(20)  NOT NULL DEFAULT 'active' COMMENT 'active / disabled',
  `created_at`    datetime     NOT NULL,
  `updated_at`    datetime     NOT NULL,
  `last_login_at` datetime     DEFAULT NULL,
  `failed_logins` int          NOT NULL DEFAULT 0 COMMENT '连续登录失败次数',
  `locked_until`  datetime     DEFAULT NULL COMMENT '锁定到期时间（防在线爆破）',
  PRIMARY KEY (`user_id`),
  UNIQUE KEY `uq_tenant_username` (`tenant_id`, `username`),
  KEY `idx_role` (`role`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='账号：密码只存 bcrypt 哈希';

CREATE TABLE IF NOT EXISTS `refresh_tokens` (
  `token_hash` char(64)     NOT NULL COMMENT 'sha256(刷新令牌)；明文只在响应里出现一次',
  `user_id`    varchar(32)  NOT NULL,
  `issued_at`  datetime     NOT NULL,
  `expires_at` datetime     NOT NULL,
  `revoked_at` datetime     DEFAULT NULL COMMENT '登出或轮换时写入',
  `user_agent` varchar(255) DEFAULT NULL COMMENT '便于"我在哪些设备登录过"',
  PRIMARY KEY (`token_hash`),
  KEY `idx_user` (`user_id`),
  KEY `idx_expires` (`expires_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='刷新令牌：只存哈希，可撤销可轮换';

CREATE TABLE IF NOT EXISTS `audit_log` (
  `id`         bigint       NOT NULL AUTO_INCREMENT,
  `created_at` datetime     NOT NULL,
  `actor_id`   varchar(32)  DEFAULT NULL COMMENT '操作者 user_id；匿名操作（注册、登录失败）为空',
  `actor_name` varchar(64)  DEFAULT NULL,
  `tenant_id`  varchar(32)  DEFAULT NULL,
  `action`     varchar(64)  NOT NULL COMMENT 'auth.register / auth.login / kb.upload / kb.publish 等',
  `target`     varchar(255) DEFAULT NULL COMMENT '被操作对象：文件名、工单号等',
  `result`     varchar(20)  NOT NULL COMMENT 'ok / denied / failed',
  `detail`     varchar(500) DEFAULT NULL COMMENT '补充信息；严禁写入密钥、完整手机号等敏感值',
  `request_id` varchar(64)  DEFAULT NULL COMMENT '与结构化日志关联',
  `client_ip`  varchar(64)  DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_created` (`created_at`),
  KEY `idx_actor` (`actor_id`),
  KEY `idx_action` (`action`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='审计日志：谁在什么时候做了什么、结果如何';

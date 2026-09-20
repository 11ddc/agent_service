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
  PRIMARY KEY (`doc_id`),
  KEY `idx_uploaded` (`uploaded_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='文档元数据：重建一致性校验 + 切分参数版本管理';

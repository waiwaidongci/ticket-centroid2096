# 002 runs 增加开跑快照

`runs` 新增 `checkpoint` JSON 列：开跑时保存预测与预计件数快照，中断恢复沿用快照汇总，不重算预测、不换预计件数。导出发布沿用暂存后替换，发布中断保留旧版。

- 全新部署由 SQLAlchemy 自动建列，无需操作。
- 已有实例需显式迁移：`ALTER TABLE runs ADD COLUMN checkpoint JSON;`
- 演示环境也可按 README 用 `docker compose down -v --remove-orphans` 清空数据卷后重建。

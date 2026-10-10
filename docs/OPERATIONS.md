# Paper Daily 运维与数据存储

## 2026-10 部署故障

10 月 6–10 日定时任务在推送时失败。10 月 9 日运行
<https://github.com/Xunmenggod/robot-paper-daily/actions/runs/37897699462>
生成的 `arxiv_cs_ro_papers_final.json` 达到 100.45 MiB，超过 GitHub 单文件
100 MiB 硬限制。摘要服务同时返回 HTTP 403，提示旧免费密钥失效；原程序捕获
错误、追加失败记录并正常退出，因而生成步骤错误地显示成功。失败论文反复追加
也会加快文件增长。

再次复核 10 月 10 日运行
<https://github.com/Xunmenggod/robot-paper-daily/actions/runs/38032168876>：
同样在推送 100.45 MiB 文件时被拒收，且有 100 次摘要 HTTP 403。

## 归档格式与历史保留

- 权威数据：`data/papers/YYYY-MM-DD.json`；每个文件保留该日期的完整记录列表。
- `manifest.json` 记录每片的 SHA-256 和记录数。缺失、损坏、多余分片均阻止运行，
  不再用空历史覆盖旧数据。保存拒绝移除历史日期或论文链接（包括重复记录的数量）。
- 单片限制为 50 MiB，提交前另查 Git 暂存区 blob 大小。超限会在推送前明确失败，
  不会删记录或静默截断。当前最大分片不到 1 MiB。
- 每日只写发生变化的日期文件及清单；无需 LFS、额外存储账号、密钥或付费服务。
- JSON 文件逐个原子替换，清单最后写入。如果本地写入被强制中断，校验会拒绝
  不一致的目录；请保留该工作目录，从最后正常 Git 提交的新副本恢复，再重新抓取。
  不要直接重算清单来掩盖数据损坏。CI 中失败的工作区不会推送。
- Markdown 与 HTML 都从该归档读取最近五个有数据的日期，停更超过五天时仍可
  离线生成最后的有效论文页面。历史数据仍可完整导出。

迁移基线：`fd90d2d8f95cd8f32398269f47f70e03e6e758f2`（2026-10-05）。
原文件 104,546,733 字节，373 个日期、12,341 条记录，范围 2025-09-24 至
2026-10-05。迁移用 Python 对旧、新数据执行完整相等断言，包含所有字段、列表
顺序和重复记录；最大新分片 861,285 字节。4,684 条原有失败摘要原样保留，
没有批量删除、重新调用模型或承诺补齐。未来遇到尚无成功摘要的已有论文时，
仅原位更新一条失败记录的摘要字段；已有重复历史不做清理。

本次只把单文件从当前 Git 树迁移为分片；没有重写历史、force push 或移除旧提交。
旧版大文件仍可从上述基线提交恢复。既有 Git 历史占用不会因本次迁移消失；长期
仓库总量仍需监控，接近 GitHub 的仓库体积建议时再评估外部对象存储，不能靠删除
历史数据解决。GitHub 限制说明：
<https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github>

## 本地离线验证

推荐 Python 3.11（与 CI 相同）：

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python paper_store.py validate
python create_index.py --output /tmp/paper-index.html
python -c 'from paper_daily import json_to_markdown; json_to_markdown("data/papers", "/tmp/paper-report.md")'
python scripts/check_staged_sizes.py --all
```

这些命令不会请求 arXiv 或摘要 API，不需要密钥。旧版 JSON 仍可显式作为 HTML
输入，例如 `python create_index.py --json /path/to/legacy.json --output /tmp/index.html`。

完整导出（仅供本地使用，不要提交回 Git）：

```sh
python paper_store.py export --output /tmp/arxiv_cs_ro_papers_final.json
```

从其他旧副本迁移时，使用新的空目标目录；命令会逐字段校验往返一致，并保留源文件：

```sh
python paper_store.py migrate --legacy /path/to/legacy.json --archive /tmp/paper-archive
```

## 摘要密钥与恢复上线

`PAPER_TOKEN` 是既有 GitHub Actions secret，映射到 `LLM_API_KEY`；
`EMBODIED_PROMPT` 是既有 repository variable。程序不再打印它们，也不记录服务商
错误响应正文；HTTP 401/403/429/5xx、网络错误及无效摘要会非零退出并阻止发布。
不会自动换服务、降级为失败摘要或重试失效凭据。

需要仓库所有者自行在服务商账户确认有效额度与授权，然后在 GitHub
Settings → Secrets and variables → Actions 更新 `PAPER_TOKEN`。不要在聊天、
提交、日志或 PR 中贴密钥。现有 `GIT_TOKEN` 继续用于发布，不在本次修改中替换。
离线 CI 通过只能证明代码和存储校验通过，不能证明在线摘要服务已恢复。

修复分支的 push/PR 仅运行 `Validate paper pipeline`，使用只读权限且无 secrets。
生产任务仅允许 main 上的定时或手动事件；同组任务串行执行，普通快进推送遇到
main 并发更新会安全失败，不会自动 rebase 或覆盖他人改动。

上线应在所有者批准合并、确认凭据有效后进行，再观察下一次正常定时运行。
本次修复不主动合并、不手动触发生产任务。10 月 6 日以后的失败运行未成功推送
新数据，近期抓取也不能保证补全整个停更区间；需要补抓时应另外确认时间范围与
模型额度。

# AGENTS.md — 工作约束

本文件定义了 Reasonix 在此项目中的工作方式。每次开始任务前，请先阅读并遵守。

---

## 1. 复杂任务必须先计划再执行

- 大于 3 步的任务必须先写计划（`todo_write`），经用户批准后再实施。
- 计划以两层级展开：Phase（level 0）→ 具体子步骤（level 1），每步完成后用 `complete_step` 签收。

## 2. 修改代码前优先使用 codegraph 理解项目

- 调用 `codegraph_context` 获取相关符号、入口点、调用链。
- 不要一开始就全量 `read_file` — 先用 codegraph 定位到精确位置。

## 3. 不要一开始全量读取文件

- 先通过 `ls` / `glob` / `codegraph_search` 了解结构，再按需读具体符号或行范围。

## 4. 只做最小必要修改

- 改什么文件、改多少行，以刚好满足需求为度。
- 优先用 `edit_file` / `multi_edit` 做精确替换，不要重写整个文件。

## 5. 修改后尽可能运行测试或检查命令

- 如果有测试套件，运行相关测试。
- 否则运行 `lsp_diagnostics` 检查语法/类型错误，或手动验证关键路径。

## 6. 涉及论文、科研、PDF、Excel、统计、可视化、数据分析时，优先使用已安装的 scientific-agent-skills

- 在 Skills 索引中查找相关 skill（如 `pdf`、`citation-management`、`scientific-writing`、`xlsx` 等），优先通过 `run_skill` 或 `slash_command` 调用。

## 7. 用户明确要求使用某个 skill 时，必须调用对应 skill

- 通过 `run_skill({ name: "<skill-name>", arguments: "<task>" })` 执行。

## 8. 完成后总结

- 每项任务完成后，用一句话总结：做了什么、改了哪些文件、是否经过验证、还有什么未解决的风险或遗留事项。

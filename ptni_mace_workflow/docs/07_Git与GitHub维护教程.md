# 07 Git 与 GitHub 维护教程

这份教程面向 Git 新手，用于维护本项目的代码、文档、版本号和 GitHub Pages。

## 核心概念

| 名词 | 这里的含义 |
| --- | --- |
| 工作区 | 你本地实际看到和编辑的文件 |
| 暂存区 | 准备放进下一次 commit 的文件清单 |
| commit | 一次带说明的本地版本快照 |
| tag | 指向某个 commit 的版本号，例如 `v0.1.1` |
| remote/origin | GitHub 上的远端仓库 |
| push | 把本地 commit/tag 上传到 GitHub |
| GitHub Pages | GitHub 自动发布的网页文档 |

## 本项目哪些文件应该提交

应该提交：

```text
ptni_mace_workflow/
.github/workflows/pages.yml
.gitignore
VERSION
CHANGELOG_中文.md
GIT_SUBMIT_COMMANDS.md
```

不应该提交：

```text
work/
mace_workspace/
outputs/
checkpoints/
logs/
results/
wandb/
*.model
*.pt
*.extxyz
```

这些大文件和运行目录已经由 `.gitignore` 排除。

## 日常修改后的固定流程

每次 Codex 或你自己修改代码/文档后，先检查：

```bash
git status --short --branch
```

查看具体改了什么：

```bash
git diff
```

如果是新文件，确认它没有被 `.gitignore` 误挡：

```bash
git status --ignored --short
```

## 新增文件或数据怎么上传

先判断这个新增内容属于哪一类。

应该上传到 GitHub 的内容：

```text
脚本
Markdown 文档
GitHub Actions 配置
小型示例输入
小型测试 fixture
轻量 CSV/JSON manifest
```

不应该上传到 GitHub 的内容：

```text
完整训练集
大 extxyz
OUTCAR/POTCAR/WAVECAR/CHGCAR
MACE .model/.pt
训练 checkpoint
benchmark 运行结果
W&B 本地日志
```

如果新增的是普通代码或文档：

```bash
git add path/to/file
```

如果新增的是一个目录：

```bash
git add path/to/directory
```

如果新增的是小型示例数据，但扩展名被 `.gitignore` 忽略，例如一个很小的 `example.extxyz`，先确认它真的适合公开上传：

```bash
du -h path/to/example.extxyz
```

然后可以强制加入：

```bash
git add -f path/to/example.extxyz
```

只对小型、脱敏、可公开的数据这样做。不要对 `work/`、`mace_workspace/`、`checkpoints/` 做 `git add -f`。

如果某一类小文件以后经常要上传，优先修改 `.gitignore` 增加白名单，而不是每次都 `git add -f`。例如：

```gitignore
*.extxyz
!ptni_mace_workflow/examples/**/*.extxyz
```

这样只有 `ptni_mace_workflow/examples/` 下面的 extxyz 示例会被允许提交。

## 提交前检查

本项目推荐至少运行：

```bash
python -m compileall -q ptni_mace_workflow
python ptni_mace_workflow/tools/build_docs_site.py --out-dir _site
```

如果改了 shell 脚本，在 WSL 中运行：

```bash
find ptni_mace_workflow -name "*.sh" -print0 | xargs -0 -n1 bash -n
```

检查暂存区不要混入大文件：

```bash
git diff --cached --name-only | grep -E '\.(model|pt|pth|ckpt|extxyz|traj|db)$|^(work|mace_workspace|checkpoints|logs|results|wandb|outputs)/'
```

如果这条命令没有输出，通常就是安全的。

## 回档和撤销

先看历史：

```bash
git log --oneline --decorate --graph -10
```

只临时查看旧版本，不改变当前文件：

```bash
git show v0.1.0:path/to/file
```

把某个文件恢复到上一个提交的版本：

```bash
git restore path/to/file
```

把某个文件恢复到指定版本：

```bash
git restore --source v0.1.0 -- path/to/file
```

恢复后需要提交这个回档：

```bash
git add path/to/file
git commit -m "v0.1.2: restore path/to/file from v0.1.0"
```

如果刚刚 commit 了但还没有 push，想修改最后一次 commit：

```bash
git add changed_file
git commit --amend
```

如果已经 push 了，不建议改历史。更安全的做法是新建一个“回退提交”：

```bash
git revert <commit_id>
```

例如：

```bash
git revert 541fd3a
```

这会生成一个新的 commit，用来撤销那个旧 commit 的改动。它适合已经公开 push 的版本。

谨慎使用：

```bash
git reset --hard <commit_id>
```

这会丢弃当前工作区改动，并把分支强行移动到旧提交。除非明确要丢弃本地修改，否则不要用。

## 提交代码

添加需要提交的文件：

```bash
git add .gitignore VERSION CHANGELOG_中文.md GIT_SUBMIT_COMMANDS.md .github ptni_mace_workflow
```

确认暂存区：

```bash
git status --short
```

提交：

```bash
git commit -m "v0.1.1: add Git maintenance guide"
```

## 创建版本 tag

```bash
git tag -a v0.1.1 -m "v0.1.1: add Git maintenance guide"
```

如果 tag 已存在但需要修正：

```bash
git tag -f -a v0.1.1 -m "v0.1.1: add Git maintenance guide"
```

## 推送到 GitHub

推荐使用 SSH：

```bash
git remote set-url origin git@github.com:populuscathayana/mace-ptni.git
git push -u origin main
git push origin v0.1.1
```

如果继续使用 HTTPS，不要在 `Password` 输入 GitHub 登录密码。要输入 Personal Access Token。

Classic PAT 至少需要：

```text
repo
workflow
```

fine-grained token 至少需要：

```text
Contents: Read and write
Actions 或 Workflows: Read and write
Metadata: Read-only
```

## GitHub Pages

GitHub 仓库设置中应选择：

```text
Settings -> Pages -> Source: GitHub Actions
```

不需要添加 verified domain。自定义域名不是必须项。

默认网页地址：

```text
https://populuscathayana.github.io/mace-ptni/
```

每次 push 到 `main` 后，只要修改触发路径包含 `ptni_mace_workflow/`、`VERSION`、`CHANGELOG_中文.md` 或 `.github/workflows/pages.yml`，GitHub Actions 会自动构建并部署网页。

## 常见状态怎么理解

`git status --short --branch` 里：

| 显示 | 含义 |
| --- | --- |
| `?? file` | 新文件，还没加入 Git |
| ` M file` | 文件被修改，但还没暂存 |
| `M  file` | 文件已暂存，准备 commit |
| `A  file` | 新文件已暂存 |
| `!! file` | 被 `.gitignore` 忽略 |

如果只看到被忽略的大目录，不用处理。

## 出错时的安全原则

- 不要运行 `git reset --hard`，除非你非常确定要丢弃所有本地改动。
- 不要把 token 写进文件，也不要把 token 放进 remote URL 后提交。
- 不要强推 `main`，除非明确知道远端历史可以被覆盖。
- 如果 Codex 无法 push，会把需要你执行的命令写入 `GIT_SUBMIT_COMMANDS.md`。

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

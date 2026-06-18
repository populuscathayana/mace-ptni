# 07 Git 与 GitHub 基本操作

这份文档只记录日常最常用流程：改完代码后如何提交、如何把新文件加入 Git、如何推送到 GitHub。

## 基本逻辑

Git 的日常流程只有四步：

```text
检查改动 -> 选择要提交的文件 -> 本地提交 commit -> 上传 push
```

本项目只把代码、脚本、文档和轻量配置上传到 GitHub。

不要上传：

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

这些已经写进 `.gitignore`，正常不会被提交。

## 每次改完代码后的固定流程

先看当前状态：

```bash
git status --short --branch
```

看具体改了什么：

```bash
git diff
```

做基本检查：

```bash
python -m compileall -q ptni_mace_workflow
python ptni_mace_workflow/tools/build_docs_site.py --out-dir _site
```

如果改了 shell 脚本，再检查 shell 语法：

```bash
find ptni_mace_workflow -name "*.sh" -print0 | xargs -0 -n1 bash -n
```

加入需要提交的文件：

```bash
git add .gitignore VERSION CHANGELOG_中文.md GIT_SUBMIT_COMMANDS.md .github ptni_mace_workflow
```

确认没有把大文件放进暂存区：

```bash
git diff --cached --name-only
```

如果这里出现 `work/`、`mace_workspace/`、`checkpoints/`、`.model`、`.pt`、`.extxyz`，先停下来检查。

提交：

```bash
git commit -m "v0.1.3: simplify Git maintenance guide"
```

打版本 tag：

```bash
git tag -a v0.1.3 -m "v0.1.3: simplify Git maintenance guide"
```

推送：

```bash
git push origin main
git push origin v0.1.3
```

## 新增文件怎么加入 Git

普通代码或文档：

```bash
git add path/to/file
```

整个新目录：

```bash
git add path/to/directory
```

如果是很小、可公开的示例数据，但被 `.gitignore` 忽略了，可以强制加入：

```bash
git add -f path/to/example_file
```

只对小型示例这样做。不要对下面这些目录或文件强制加入：

```text
work/
mace_workspace/
checkpoints/
*.model
*.pt
*.extxyz
```

## 推送认证

推荐 SSH：

```bash
git remote set-url origin git@github.com:populuscathayana/mace-ptni.git
git push origin main
git push origin v0.1.3
```

如果用 HTTPS，GitHub 的 `Password` 位置不能填登录密码，必须填 Personal Access Token。

## GitHub Pages

你已经设置：

```text
Settings -> Pages -> Source: GitHub Actions
```

以后每次 push 到 `main`，GitHub Actions 会自动更新网页。

网页地址：

```text
https://populuscathayana.github.io/mace-ptni/
```

# Git 提交 / 推送指令

更新时间：2026-06-22

## 当前状态

本地仓库已经完成或即将完成：

- 分支：`main`
- 当前计划版本：`v0.2.0`
- `v0.1.0`：模块化 PtNi MACE workflow + GitHub Pages 基础部署
- `v0.1.1`：新增 Git/GitHub 维护教程，并忽略旧 `outputs/` 本地备份
- `v0.1.2`：扩展新增文件/小型数据上传教程和回档教程，清理本地可再生成缓存
- `v0.1.3`：精简 Git/GitHub 维护教程，只保留日常提交和新增文件上传流程
- `v0.1.4`：训练入口支持显式 `--epochs` 和 `--patience` 参数
- `v0.2.0`：新增 PtNi slab 顶部可动原子距离稳定性 benchmark
- 远端：`git@github.com:populuscathayana/mace-ptni.git`
- 本地最新提交：`v0.2.0: add slab distance scan benchmark`

当前 Codex 侧推送失败原因：

```text
Host key verification failed.
fatal: Could not read from remote repository.
```

这通常表示当前 shell 的 `~/.ssh/known_hosts` 里还没有 GitHub 的 host key。你可以在自己的 WSL/PowerShell 中先执行一次 `ssh -T git@github.com`，按提示输入 `yes` 信任 GitHub，然后再 push。

## 推荐推送命令

在 PowerShell 或 WSL 中进入项目目录：

```bash
cd "/mnt/c/Users/A/Documents/Codex/2026-06-08/vasp-dft-mace-outcar-slabd-ptni"
```

如果在 Windows PowerShell 中：

```powershell
cd "C:\Users\A\Documents\Codex\2026-06-08\vasp-dft-mace-outcar-slabd-ptni"
```

检查状态：

```bash
git status --short --branch
git log --oneline --decorate -3
git remote -v
```

如果已经配置好认证，推送 main 和版本 tag：

```bash
git push -u origin main
git push origin v0.2.0
```

如果 GitHub 提示 `v0.2.0` tag 已存在但指向不一致，再执行：

```bash
git push -f origin v0.2.0
```

## 推荐方式：SSH 推送

在 WSL 中检查是否已有 SSH key：

```bash
ls ~/.ssh/*.pub
```

如果没有，创建一个新的 ed25519 key：

```bash
ssh-keygen -t ed25519 -C "populuscathayana@gmail.com"
```

一路回车即可使用默认路径。然后启动 agent 并添加 key：

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

显示公钥：

```bash
cat ~/.ssh/id_ed25519.pub
```

复制整行公钥，添加到 GitHub：

```text
GitHub -> Settings -> SSH and GPG keys -> New SSH key
```

测试 SSH：

```bash
ssh -T git@github.com
```

第一次会询问是否信任 GitHub host，输入：

```text
yes
```

看到类似下面的信息即可：

```text
Hi populuscathayana! You've successfully authenticated, but GitHub does not provide shell access.
```

切换远端到 SSH 并推送：

```bash
git remote set-url origin git@github.com:populuscathayana/mace-ptni.git
git push -u origin main
git push origin v0.2.0
```

## 备用方式：HTTPS + Personal Access Token

如果继续使用 HTTPS，不要输入 GitHub 登录密码。`Username` 输入：

```text
populuscathayana
```

`Password` 位置粘贴 GitHub Personal Access Token。

Classic token 需要至少包含：

```text
repo
workflow
```

`workflow` 是因为本仓库要推送 `.github/workflows/pages.yml`。

生成 token 后保持 HTTPS remote：

```bash
git remote set-url origin https://github.com/populuscathayana/mace-ptni.git
git push -u origin main
git push origin v0.2.0
```

如果你使用的是 fine-grained token，请给 `populuscathayana/mace-ptni` 仓库至少开启：

```text
Contents: Read and write
Metadata: Read-only
Actions 或 Workflows: Read and write
```

不同 GitHub 页面版本可能显示为 `Actions` 或 `Workflows`。核心是允许推送 `.github/workflows/pages.yml`。

## 备用方式：GitHub CLI

如果安装了 GitHub CLI：

```bash
gh auth login
gh auth setup-git
git push -u origin main
git push origin v0.2.0
```

## 推送后检查

```bash
git ls-remote --heads origin main
git ls-remote --tags origin v0.2.0
```

远端 `main` 应指向本地最新提交或其后续提交。

## GitHub Pages 设置

进入 GitHub 仓库：

```text
https://github.com/populuscathayana/mace-ptni
```

打开：

```text
Settings -> Pages
```

用户已完成：

```text
Source: GitHub Actions
```

不需要添加 verified domain。`Add a verified domain` 只用于你要绑定自己的自定义域名，例如 `example.com` 或 `docs.example.com`。

本项目使用 GitHub 默认 Pages 地址即可：

```text
https://populuscathayana.github.io/mace-ptni/
```

如果 GitHub 页面问：

```text
What domain would you like to add?
```

请不要填写，返回仓库的 Pages 设置页，选择 `GitHub Actions` 作为 Source。

## 以后约定

之后如果 Codex 不能直接完成 push，会把需要你执行的 Git 指令继续写入本文件或同类本地指令文件中。

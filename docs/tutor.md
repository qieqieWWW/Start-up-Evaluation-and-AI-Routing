添加了makefile
现在无需指定配置文件路径，clone仓库后执行：
        make pre-commit-install   #安装hook到本地仓库

        make pre-commit-run  #手动运行检查

但由于windows原生不自带make工具，首次使用时需要配置
    Step 1. 以管理员身份打开powershell，执行以下内容安装Chocolatey：
        Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    Step 2. 安装make：
        choco install make
    重新打开终端即可执行

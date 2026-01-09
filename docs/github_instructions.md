# 这个文件旨在帮助组员快速了解git操作，以便尽快进入工作流程
    github就是一个适合很多人一起用，协同工作和开发的网盘
## 常用指令
    git add <位置>         #git add scripts/ 即添加整个scripts文件夹和里面和原来不一样的文件；git add scripts/data_process.py即精确到文件；git add . 将提交整个本地仓库文件夹内所有修改过的文件

    git commit -m "注释"   #把上面选中的文件加入提交列表，并为这一次修改加上一个描述

    git pull #可以拉取某个分支的更新内容到本地当前分支，我目前不是很熟悉之后再加上

    git push #把git commit的东西正式推送到远端仓库，执行完这个以后正常的话就可以在浏览器那边刷新github的仓库页面看到修改了

    git status  #用来查看你现在选中、commit了啥，查看本地状态的
## 刚开始
    先从https://git-scm.com/install/windows 下载git工具（记得在安装包的第二页(select components)勾上"Add a Git bash profile to windows terminal），然后在终端（cmd）中切换到需要存放仓库的位置
    -首次使用可能需要配置git用户名和邮箱：
        git config --global user.name "用户名"     #这个用户名无需与github账号的一致，只是作为用户标识，不是登录操作，可以随便写喜欢的
        git config --global user.email "邮箱"

    -为了保证仓库整洁，首次使用先去看docs/tutor.md，安装pre-commit工具，之后commit的时候会自动执行
### 现在可以开始git流程
    -使用git clone克隆仓库到当前目录（就是下载）：
        git clone https://github.com/qieqieWWW/Start-up-Evaluation-and-AI-Routing.git
    你现在可以在文件管理器内看到下载的仓库

    -编写代码或markdown等等新增文件前，先切换到dev分支：
        git checkout dev
    -现在拉取远端仓库*里dev分支的最新代码
        git pull origin dev
    -得到最新已存在代码后切换到一个新的你自己将要开始写的实现分支，比如叫做feature分支：
        git checkout -b feature/new-script
    -现在可以开始在scripts下新建文件开始写代码。如果你使用类似于github copilot、kiro、trae之类的ai aigent IDE写代码，他们极大概率会在工作空间内直接随意创建一些测试脚本方便他们了解代码状况，这会导致工作空间乱掉，你再push上来就直接炸了，所以我建议你在使用vibe coding（就是ide内置ai写代码）前，先给他们创建一个临时的文件夹，然后跟他们说好“除了你创建的脚本文件，他们创建的测试脚本全部要在那个临时文件夹里创建，只有正式的脚本需要在你许可后在scripts内创建”之类的约束条件。之后你可以在同步你的代码前删掉那个测试的文件夹，或者在.gitignore里加上那个文件夹来防止被github跟踪。此外，如果你觉得某个测试文件有价值，单独提交那个文件到

    -写完代码后，就该提交到仓库了。
        git add scripts/新的代码.py  #可以先这样指定到你添加了的代码，防止选中了一些别的暂时不打算push的东西
    -接下来提交选中的文件
        git commit -m "完成了xxx部分的脚本"
    -先别急着push，再切换到dev分支，检查你在写代码的过程中别的组员有没有push新的代码，避免等会合并的时候产生冲突
        git checkout dev
        git pull origin dev
    -切换回你的feature分支，把新的代码合并到dev分支
        git checkout feature/new-script
        git merge dev
    -现在可以推送了，远端会自动创建同名分支
        git push origin feature/new-script
    -接下来在网页端github我们的仓库页面点击pull requests，然后选new pull requests，选择base分支为dev，compare分支为你的feature/new-script，填写一下pr描述，写一下根据啥新增了什么之类的说明白你那个是啥玩意就行，最后点create pull request就可以了

### 一些我在创建的时候遇到的可能会造成困惑的地方
    -origin是什么玩意为啥有些指令有有些指令没有

    -分支是干啥的

    -写代码后为了确保我写代码的过程中没有新的文件加入造成等会合并到dev的时候产生冲突，可以pull一次dev分支检查一下，但是写代码前为啥也要pull一次，何意味

    -.gitignore是干啥的

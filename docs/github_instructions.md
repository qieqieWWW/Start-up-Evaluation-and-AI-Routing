# 这个文件旨在帮助组员快速了解git操作，以便尽快进入工作流程
    github就是一个适合很多人一起用，协同工作和开发的网盘
## 常用指令
    git add <位置>         #git add scripts/ 即添加整个scripts文件夹和里面和原来不一样的文件；
                          #git add scripts/data_process.py即精确到文件；git add . 将提交整个本地仓库文件夹内所有修改过的文件

    git commit -m "注释"   #把上面选中的文件加入提交列表，并为这一次修改加上一个描述

    git pull #可以拉取某个分支的更新内容到本地当前分支，我目前不是很熟悉之后再加上

    git push #把git commit的东西正式推送到远端仓库，执行完这个以后正常的话就可以在浏览器那边刷新github的仓库页面看到修改了

    git status  #用来查看你现在选中、commit了啥，查看本地状态的
## 刚开始
    先从https://git-scm.com/install/windows 下载git工具（记得在安装包的第二页(select components)勾上"Add a Git bash profile to windows terminal），
    然后在终端（cmd）中切换到需要存放仓库的位置

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
    -现在可以开始在scripts下新建文件开始写代码。如果你使用类似于github copilot、kiro、trae之类的ai aigent IDE写代码，
    他们极大概率会在工作空间内直接随意创建一些测试脚本方便他们了解代码状况，这会导致工作空间乱掉，
    你再push上来就直接炸了，所以我建议你在使用vibe coding（就是ide内置ai写代码）前，
    先给他们创建一个临时的文件夹，然后跟他们说好“除了你创建的脚本文件，他们创建的测试脚本全部要在那个临时文件夹里创建，
    只有正式的脚本需要在你许可后在scripts内创建”之类的约束条件。
    之后你可以在同步你的代码前删掉那个测试的文件夹，或者在.gitignore里加上那个文件夹来防止被github跟踪。

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
    -接下来在网页端github我们的仓库页面点击pull requests，然后选new pull requests，选择base分支为dev，compare分支为你的feature/new-script，
    填写一下pr描述，写一下根据啥新增了什么之类的说明白你那个是啥玩意就行，最后点create pull request就可以了
    -如果你觉得合并后你的分支没用了，使用
        git branch -d 分支名        #删除本地分支
        git push origin --delete 分支名   #删除远程分支

### 一些我在创建的时候遇到的可能会造成困惑的地方
    Q1. origin是什么玩意为啥有些指令有有些指令没有
    A：origin用来指代完整的仓库url，是一个默认的约定，在初始化仓库的时候就会设置：git remote add origin https://github.com/xxxxx.git 那个origin可以改成想要的名字的。
    需要带origin的指令是涉及远程仓库操作的指令，比如git push origin; git pull origin，都是向远程仓库发起请求的操作，而那些本地操作比如git add/commit，就不需要带url/origin。

    Q2. 分支是干啥的
    A：用来隔离仍在试验的代码和已经完成编写可用的代码，比如main分支就是默认主分支，一般用来存放已经可用的代码，
    而功能分支（feature之类的）就是用来试验的，在里面修修改改都没问题，等稳定了再请求合并到主分支就完成了一个功能的开发。
    如果所有人都在main里面改，就相当于所有人在同一个文件夹里写自己的东西，很难不乱，万一有人写了一半的东西不小心push了就麻烦了。
    因为每个人开发新功能都需要互不干扰，而全部合并后新的代码对原来的代码这个整体来说又是新的东西，
    所以我们的流程应该是自己在编写的时候创建自己的分支，然后合并到dev分支，最后再合并到main就最稳定。


    Q3. 写代码后为了确保我写代码的过程中没有新的文件加入造成等会合并到dev的时候产生冲突，可以pull一次dev分支检查一下，但是写代码前为啥也要pull一次，何意味
    A: 写代码前同步一次dev分支是为了保证你即将开始编写的内容是基于最新内容的，比如有个你要调用的function在最新的版本中被修改了，
    但你本地的版本还是没改的，那你写的就会和新的不兼容。
    提交前再pull一次是为了防止其他组员提交了新的代码到dev，导致可能出现的合并冲突。

    Q4. .gitignore是干啥的
    A：就是一个黑名单，里面写上你不想要跟踪的文件或文件夹或文件类型，使用git add .的时候它们就不会被加入暂存区，
    你可以用git status检查准备提交的文件有啥。

    Q5. 我不小心git add了错误的文件/文件列表，git status里显示它们都在暂存区了，咋办？
    A：分几种可能的场景：
            -暂存区里大部分是正确的需要被提交的文件，只有某个、某两个文件/文件夹是不打算提交的，并且以后也不打算提交
            比如仓库文件夹同时也是你的工作空间，某个文件夹只是你自己创建用来放自己的笔记或者大家发在群里的pdf什么的
            直接去.gitignore加上文件夹名，保存后哪个文件夹在git status里会消失，如果还有就再git add一次相同目录即可

            -暂存区里大部分是正确的需要被提交的文件，只有某个、某两个文件/文件夹是不打算提交的，但不想把它添加到.gitignore
            有计划以后提交到仓库
            用git reset HEAD 文件夹/文件 来精准删除暂存区里的文件

            -暂存区里全错了，我不小心在错误的目录下使用了git add .
            用git reset HEAD
            执行后git status里to be commit 就会变成Changes not staged for commit，即暂存区空了

    Q6. 既然是网盘，有没有空间限额？
    A: 有应该是有，不过我不知道，但github的仓库不允许超过100m的文件被push，那些小小的代码文件也摸不到顶，提交前记得在
    .gitignore里加上大文件。你下载了原始数据集在你的本地运行data_process.py处理后，脚本会把你的原始数据集放到datasets/里，
    创建的清洗后数据集会有专门的文件夹存放，也就是会自动规范结构，而gitignore里已经有了这两个文件夹，所以一般来说你是不会遇到
    大文件报错的情况的。

# Developer guide for fastai
> Making your first pull request


In order to contribute to fastai (or any fast.ai library... or indeed most open source software!) you'll need to make a [pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests), also known as a *PR*. Here's an [example](https://github.com/fastai/fastai/pull/2648) of a pull request. In this case, you can see from the description that it's something that fixes some typos in the library. If you click on "Files changed" on that page, you can see all the changes made. We get notified when a pull request arrives, and after checking whether the changes look OK, we "merge" it (which means that we click a button in GitHub that causes all those changes to get automatically added to the repo).

Making a pull request for the first time can feel a bit over-whelming, so I've put together this guide to help you get started. We're going to use GitHub's command line tool `gh`, which makes things faster and easier than doing things through the web-site (in my opinion, at least!)

I'm assuming in this guide that you're using Linux, and that you've already got [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) set up. This is the default in all of the fast.ai course guides, and is highly recommended.
> This document is created from a Jupyter Notebook. You can run the notebook yourself by getting it [from here](https://github.com/fastai/fastai/blob/master/nbs/dev-setup.ipynb). You'll need to install the [Jupyter Bash kernel](https://github.com/takluyver/bash_kernel).

## One time only setup

### Setting up access and `gh`

Install `fastai`, `gh`, and `nbdev` (this also checks you have the latest versions, if they're already installed):

```python
conda install -y -c fastai -c pytorch -c anaconda anaconda fastai gh nbdev
```



**NB**: if you're using miniconda instead of Anaconda, remove `-c anaconda anaconda` from the above command.

You'll need to set up `ssh` access to GitHub, if you haven't already. To do so, follow [these steps](https://docs.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account). Once you've created an ssh key (generally by running `ssh-keygen`), you can copy the contents of your `.~/ssh/id_rsa.pub` file and paste them into GitHub by clicking "New SSH Key" on [this page](https://github.com/settings/keys).

Once that's working, we need to get a [personal access token](https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token) to allow `gh` to connect to GitHub. To do so, [click here](https://github.com/settings/tokens/new) and enter "gh" in the "Note" section, and click the `repo`, `read:discussion`, and `read:org` checkboxes (note that `gh` can do this automatically for you, but it only works conveniently if you're running the code on your local machine; most fastai developers are probably using a remote GPU server, such as Paperspace, AWS, or GCP, so we show the approach below because it works for everyone).

{% include image.html alt="Personal access token screen" caption="Personal access token screen" max-width="495" file="/images/att_00000.png" %}

Then click "Generate Token" at the bottom of the screen, and copy the token (the long string of letters and numbers shown). You can easily do that by clicking the little clipboard icon next to the token.

{% include image.html alt="Copying your token" caption="Copying your token" max-width="743" file="/images/att_00001.png" %}

Now run this in your shell, replacing `jph01` with your GitHub username, and the string after `TOKEN=` with your copied token:

```python
GH_USER=jph01
TOKEN=abae9e225efcf319f41c68f3f4d7c2d92f59403e
```

Setup `gh` to use `ssh` to connect to GitHub, so you don't have to enter you name and password all the time:

```python
gh config set git_protocol ssh
```

Create your GitHub authentication file:

```python
echo -e "github.com:\n  user: $GH_USER\n  oauth_token: $TOKEN\n" > ~/.config/gh/hosts.yml
```

### Set up `fastcore`

Now we're ready to clone the `fastcore` and `fastai` libraries. We recommend cloning both, since you might need to make changes in, or debug, the `fastcore` library that `fastai` uses heavily. First, we'll do `fastcore`:

```python
gh repo clone fastai/fastcore
```

    Cloning into 'fastcore'...
    remote: Enumerating objects: 247, done.[K
    remote: Counting objects: 100% (247/247), done.[K
    remote: Compressing objects: 100% (164/164), done.[K
    remote: Total 1618 (delta 162), reused 139 (delta 78), pack-reused 1371[K
    Receiving objects: 100% (1618/1618), 3.18 MiB | 18.61 MiB/s, done.
    Resolving deltas: 100% (1102/1102), done.


We update our installed version to use [editable mode](https://stackoverflow.com/questions/35064426/when-would-the-e-editable-option-be-useful-with-pip-install). This means that any changes you make to the code in your checked-out repo will automatically be used everywhere on your computer that uses this library:

```python
cd fastcore
pip install -qe .
```

We update the repo to create and use a [fork](https://medium.com/@tharis63/git-fork-vs-git-clone-8aad0c0e38c0#:~:text=Git%20Fork%20means%20you%20just,to%20your%20own%20GitHub%20profile.&text=Then%20make%20your%20changes%20and,it%20to%20the%20main%20repository.):

```python
gh repo fork --remote
```

    [K[90m- [90mForking [0m[1;39m[90mfastai/fastcore[0m[0m[90m...[0m[0m
    [0m[33m![0m [1;39mjph01/fastcore[0m already exists
    [32m✓[0m Renamed [1;39morigin[0m remote to [1;39mupstream[0m
    [32m✓[0m Added remote [1;39morigin[0m


Because all fast.ai libraries use nbdev, we need to run [nbdev_install_git_hooks](https://nbdev.fast.ai/cli#Git-hooks) the first time after we clone a repo; this ensures that our notebooks are automatically cleaned and trusted whenever we push to GitHub:

```python
nbdev_install_git_hooks 
```

    Executing: git config --local include.path ../.gitconfig
    Success: hooks are installed and repo's .gitconfig is now trusted


### Set up `fastai`

Now we'll do the same steps for `fastai`. Since `fastai` includes submodules (for the docs), we need to set those up too:

```python
cd ..
gh repo clone fastai/fastai
cd fastai
```

    Cloning into 'fastai'...
    remote: Enumerating objects: 108, done.[K
    remote: Counting objects: 100% (108/108), done.[K
    remote: Compressing objects: 100% (79/79), done.[K
    remote: Total 10206 (delta 41), reused 57 (delta 27), pack-reused 10098[K
    Receiving objects: 100% (10206/10206), 536.85 MiB | 38.67 MiB/s, done.
    Resolving deltas: 100% (8053/8053), done.
    Submodule 'docs' (https://github.com/fastai/fastai-docs.git) registered for path 'docs'
    Cloning into '/home/jph01/gt/fastai/docs'...
    Submodule path 'docs': checked out '3f7f20ce745e36b3b93d55a0640d94866060bd46'
    Already up to date.


We'll do an editable install of `fastai` too:

```python
pip install -qe .[dev]
```

...and fork it and install the git hooks:

```python
gh repo fork --remote
```

    [K[90m- [90mForking [0m[1;39m[90mfastai/fastai[0m[0m[90m...[0m[0m
    [0m[33m![0m [1;39mjph01/fastai[0m already exists
    [32m✓[0m Renamed [1;39morigin[0m remote to [1;39mupstream[0m
    [32m✓[0m Added remote [1;39morigin[0m


```python
nbdev_install_git_hooks 
```

    Executing: git config --local include.path ../.gitconfig
    Success: hooks are installed and repo's .gitconfig is now trusted


### Submodules

If the repo you are cloning has any submodules (fastai has [`docs`](https://fastcore.fast.ai/foundation#docs) as a submodule), initialize them too:

```python
git submodule init
git submodule update
```

We also fork the submodule, and ensure it's up to date and has `master` checked out:

```python
cd docs
gh repo fork --remote
git checkout master
git pull
```

    [K[90m- [90mForking [0m[1;39m[90mfastai/fastai-docs[0m[0m[90m...[0m[0
    [0m[33m![0m [1;39mjph01/fastai-docs[0m already exists
    [32m✓[0m Renamed [1;39morigin[0m remote to [1;39mupstream[0m
    [32m✓[0m Added remote [1;39morigin[0m
    Previous HEAD position was 3f7f20c docs
    Switched to branch 'master'
    Your branch is up to date with 'upstream/master'.
    Already up to date.


Now we're ready to do our PR. Let's `cd` back to the main `fastai` directory first:

```python
cd ..
```

## Creating your PR

Everything above needs to be done just once. From here on are the commands to actually create your PR.

Create a new [git branch](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging), by running the following, replacing `test-pr` with the name you want to give your pull request (use something that will be easy for you to remember in the future if you need to update your PR):

```python
git checkout -b test-pr
```

    M	docs
    Switched to a new branch 'test-pr'


Make whatever changes you want to make in the notebooks, and remember to run [`nbdev_build_lib`](https://nbdev.fast.ai/cli#nbdev_build_lib) when you're done to ensure that the libraries are built from your notebook changes (unless you only changed markdown, in which case that's not needed). It's also a good idea to check the output of `git diff` to ensure that you haven't accidentally made more changes than you planned.

When you're ready, `commit` your work, replacing "just testing" here with a clear description of what you did in your commit:

```python
git commit -am "just testing"
```

    [test-pr 3397cc9] just testing
     2 files changed, 2 insertions(+), 1 deletion(-)


The first time you push from your fork, you need to add `-u origin HEAD`, but after the first time, you can just use `git push`.

```python
git push -u origin HEAD
```

    Counting objects: 4, done.
    Delta compression using up to 64 threads.
    Compressing objects: 100% (4/4), done.
    Writing objects: 100% (4/4), 383 bytes | 383.00 KiB/s, done.
    Total 4 (delta 3), reused 0 (delta 0)
    remote: Resolving deltas: 100% (3/3), completed with 3 local objects.[K
    remote: 
    remote: Create a pull request for 'test-pr' on GitHub by visiting:[K
    remote:      https://github.com/jph01/fastai/pull/new/test-pr[K
    remote: 
    To github.com:jph01/fastai.git
     * [new branch]      HEAD -> test-pr
    Branch 'test-pr' set up to track remote branch 'test-pr' from 'origin'.


Now you're ready to create your PR. To use the information from your commit message as the PR title, just run:

```python
gh pr create -f
```

    https://github.com/fastai/fastai/pull/2664


To be interactively prompted for more information (including opening your editor to let you fill in a detailed description), just run `gh pr create` without the `-f` flag. As you see above, after it's done, it prints the URL of your new PR - congratulations, and thank you for your contribution!

{% include image.html alt="The completed pull request" caption="The completed pull request" max-width="615" file="/images/att_00002.png" %}

## Post-PR steps

To keep your fork up to date with the changes to the main fastai repo, and to change from your `test-pr` branch back to master, run:

```python
git pull upstream master
git checkout master
```

    From github.com:fastai/fastai
     * branch            master     -> FETCH_HEAD
    Already up to date.
    M	docs
    Switched to branch 'master'
    Your branch is up to date with 'upstream/master'.


In the future, once your PR has been merged or rejected, you can delete your branch if you don't need it any more:

```python
git branch -d test-pr
```

    Deleted branch test-pr (was 514782a).
    (base) 



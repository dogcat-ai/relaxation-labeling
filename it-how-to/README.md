This is a general IT how-to document, and the methods described here do not necessarily pertain specifically to this repo.
# Git Large File Storage (Git LFS)
I.  Go to https://git-lfs.github.com/ .
II. Follow the instructions therein.
    A. Download and install.
        1. Click download (currently points to https://github.com/git-lfs/git-lfs/releases/download/v3.0.1/git-lfs-linux-amd64-v3.0.1.tar.gz).
        2. Run `git lfs install` or unzip the tar ball and execute `install.sh`.
    B. Track the files you want to track, from inside the repos wherein you want to track 'em.
        1. For example, `git lfs track "*.png"`
            a. Don't forget to `git add .gitattributes`.
    C. There is no step 3/step C.
        1. Just add/commit/push as you normally would.
        

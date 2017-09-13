# models
Repository hosting Kipoi models

## TODO

### Contributing

- we need some sort of version control at the beginning
  - pull requests...

- [ ] explore the pull push possibilities
  - via pull requiest
    - since we are hosting everything on git lfs
      - How to prevent from downloading the whole model zoo when contributing/pulling models?
		- [x] google: git pull only a subpart of the repository (one specific folder)
	      - not possible for saving disk space: https://stackoverflow.com/a/13738951
   		    - the repo will still get loaded into .git/
  		  - proposed solutions:
			- https://lakehanne.github.io/git-sparse-checkout
			- https://stackoverflow.com/questions/600079/how-do-i-clone-a-subdirectory-only-of-a-git-repository
        - [ ] google: git pull everything except the git lfs stuff
          - 
        - [ ] google: make a pull request for only one folder?
	  - [x] git lfs track
	    - can't track by size: https://github.com/git-lfs/git-lfs/issues/282
		  - what would be the asterix?
	- how to automatically issue a pull request?
      - models push?

- [ ] shall the folder structure be: <github username>__model?


## Docs

- put all the non-code files (serialized models, test data) into a `*files` directory, where `*` can be anything.
  - Examples:
    - `model_files`
	- `dataloader_files`

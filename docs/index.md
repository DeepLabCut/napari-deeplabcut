# Welcome to napari-deeplabcut

For documentation, please see: https://deeplabcut.github.io/DeepLabCut/docs/napari_GUI.html

## Maintenance

### Testing

The package can be locally tested running `python -m pytest`. Note that these tests are automatically triggered anyway by pull requests targeting the main branch or push events.

### Deployment

Versioning and deployment builds upon napari's plugin cookiecutter setup
https://github.com/napari/cookiecutter-napari-plugin/blob/main/README.md#automatic-deployment-and-version-management.
In short, it suffices to create a tagged commit and push it to GitHub; once the `deploy` step of the GitHub's CI workflow has completed, the package has been automatically published to PyPi.

```bash
# the tag will be used as the version string for the package
git tag -a v0.1.0 -m "v0.1.0"

# make sure to use follow-tags so that the tag also gets pushed to github
git push --follow-tags
# alternatively, you can set follow-tags as default with git config --global push.followTags true
```

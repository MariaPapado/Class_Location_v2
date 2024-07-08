# Class Location V2



## Journey so far

- create_data_masks.py \
  Used it to create building masks from the geojson polygons of Alexander. For NatGrid and NatFuel \
  This is a dataset for training.

- then i trained the mamba model (only the building detection branch) \
   find repo in subfolder ./mamba_train \
   .............

- pipeline.py \
  (so far we did it on images from ale's pickle files [testing phase]) \
  to get the class location predictions

- overlapping_polygons.py \
  filter appearing buildings that already exist in the street map

- write2system.py \
  pass predicted polygons to the system

(#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.orbitaleye.nl/maria.papadomanolaki/Class_Location_v2.git
git branch -M main
git push -uf origin main
```

# %% 
import pandas as pd
path = 'C:/Users/Sabrina/Desktop/Horses-Byron-2019-05-08/training-datasets/iteration-0/UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5'
path_save = 'C:/Users/Sabrina/Documents/Horses-Byron-2019-05-08/training-datasets/iteration-0/UnaugmentedDataSet_HorsesMay8/CollectedData_Byron.h5'
df = pd.read_hdf(path)
df_save = pd.read_hdf(path_save)

print(df.shape)
print(df_save.shape ) #27 dif!
# %%
import numpy 
# %%
#Find Rows Which Are Not common Between Two dataframes, 10?
pd.concat([df,df_save]).drop_duplicates(keep=False)








# %%
#Find Rows in DF1 Which Are Not Available in DF2
df_new = df_save.merge(df, how = 'left' ,indicator=True).loc[lambda x : x['_merge']=='both']

df_new


# %%
from napari_deeplabcut import misc
import os
import numpy as np
# %%
temp = df.copy()
header = misc.DLCHeader(df.columns)
#header
temp = temp.droplevel("scorer", axis=1)
if "individuals" not in temp.columns.names:
    # Append a fake level to the MultiIndex
    # to make it look like a multi-animal DataFrame
    old_idx = temp.columns.to_frame()
    old_idx.insert(0, "individuals", "")
    temp.columns = pd.MultiIndex.from_frame(old_idx)
if isinstance(temp.index, pd.MultiIndex):
    temp.index = [os.path.join(*row) for row in temp.index]
df1 = (
    temp.stack(["individuals", "bodyparts"])
    .reindex(header.individuals, level="individuals")
    .reindex(header.bodyparts, level="bodyparts")
    .reset_index()
)
nrows = df1.shape[0]
data = np.empty((nrows, 3))
image_paths = df1["level_0"]
# %%
nrows = df.shape[0]
data = np.empty((nrows, 3))
image_paths = df["level_0"]
if np.issubdtype(image_paths.dtype, np.number):
    image_inds = image_paths.values
    paths2inds = []
else:
    image_inds, paths2inds = misc.encode_categories(
        image_paths,
        return_map=True,
    )
data[:, 0] = image_inds
data[:, 1:] = df[["y", "x"]].to_numpy()
# %%
metadata = _populate_metadata(
    header,
    labels=df["bodyparts"],
    ids=df["individuals"],
    likelihood=df.get("likelihood"),
    paths=list(paths2inds),
)
metadata["name"] = os.path.split(filename)[1].split(".")[0]
metadata["metadata"]["root"] = os.path.split(filename)[0]
layers.append((data, metadata, "points"))
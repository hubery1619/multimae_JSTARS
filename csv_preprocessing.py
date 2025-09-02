import pandas as pd

# load csv file
df = pd.read_csv('dataset/fmow-sentinel/val.csv')

# delete the column 0
df.drop(df.columns[0], axis=1, inplace=True)

# construct the column of "image_path"
image_path = df.apply(lambda row: f"dataset/fmow-sentinel/val/{row['category']}/{row['category']}_{row['location_id']}/{row['category']}_{row['location_id']}_{row['image_id']}.tif", axis=1)

# new a DataFrame and set the "image_path" as the first column
df_new = pd.concat([image_path.rename('image_path'), df], axis=1)

# save the csv file
df_new.to_csv('dataset/fmow-sentinel/val_updated.csv', index=False)

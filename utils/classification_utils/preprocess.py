import cudf, os
from sklearn.model_selection import train_test_split

def image_map(image_dir):
    image_map = {}
    for fname in os.listdir(image_dir):
        if fname.endswith(".jpg"):
            try:
                img_id = int(fname.split("_")[1].split(".")[0])
                image_map[img_id] = os.path.join(image_dir, fname)
            except (IndexError, ValueError):
                continue
    
    mapping_df = cudf.DataFrame({
        'id': list(image_map.keys()), 
        'image_path': list(image_map.values())
    })
    return mapping_df

def prepare_data(annotation_path="./datasets/SkinCAP/skincap_v240623.csv", 
                 image_dir="./datasets/images"):
    df = cudf.read_csv(annotation_path)
    print(f"Loaded {len(df)} records")

    mapping_df = image_map(image_dir)
    df = df.merge(mapping_df, on='id', how='inner')
    print(f"After attaching images: {len(df)} records")

    pandas_df = df.to_pandas()
    pandas_df = pandas_df[pandas_df['disease'].map(pandas_df['disease'].value_counts()) > 1]

    train_df, val_df = train_test_split(
        pandas_df, test_size=0.2, random_state=42, stratify=pandas_df["disease"]
    )
    
    return cudf.from_pandas(train_df), cudf.from_pandas(val_df)

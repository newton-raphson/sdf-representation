def load_data(data_path,config):

    # check if the csv exists in the path or not
    df_uniform_points = pd.read_csv(os.path.join(save_directory,"uniform.csv"))
    df_on_surface = pd.read_csv(os.path.join(save_directory,"surface.csv"))
    df_narrow_band = pd.read_csv(os.path.join(save_directory,"narrow.csv"))
    columns = df_uniform_points.columns
    df_additional = pd.DataFrame(columns=columns)
    if config.mismatchuse:
        df_additional = pd.read_csv(os.path.join(data_path,"mismatch.csv"))
    
    # print(f"length of df mismatch is {len(df_mismatch)}")
    # Create a list of data frames to concatenate, subject to the condition
    dfs_to_concat = [df for df in [df_uniform_points, df_on_surface, df_narrow_band,df_mismatch,df_filtered] if len(df) > 1]
    only_surface=False
    if len(df_on_surface)>1 and (len(df_narrow_band)<=1 and len(df_uniform_points)<=1):
       only_surface=True
    # Concatenate the data frames in the list if there are more than one
    df = pd.concat(dfs_to_concat, ignore_index=True)
    total_points = len(df)
    print(f"total_points are {len(df)}")
    feature_columns = ['x','y','z']
    target_column = ['S','nx','ny','nz']
    random_seed=227
    X_train, val_X, y_train, val_Y = train_test_split(df[feature_columns], df[target_column],test_size=0.1, random_state=random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    Y = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    # val_X = torch.tensor(val_X[0].values, dtype=torch.float32).to(0)
    # val_Y = torch.tensor(val_Y[0].values, dtype=torch.float32).to(0)
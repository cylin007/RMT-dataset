"""### Loading Data"""

batch_size = args.batch_size
valid_batch_size = args.batch_size
test_batch_size = args.batch_size
data = {}

feature_id = 0
for range_type in ['','ma3','ma6','ga12','ga24']:
    print("range_type", range_type)
    for category in ['train', 'val', 'test']:

        if range_type == "":
          category = category
          key = category
        else:
 
          key = category + "_" + str(feature_id)
          category = category + "_" + range_type

        # Loading npz
        cat_data = np.load(os.path.join(args.data, category + '.npz'))
        print("loading:", category ,'->', args.data, category + '.npz')

        data['x_' + key] = cat_data['x'][:]     # (?, 12, 207, 2)
        data['y_' + key] = cat_data['y'][:]   # (?, 12, 207, 2)

    if range_type == '':
        # 使用train的mean/std來正規化valid/test #
        scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
        data['scaler'] = scaler

    print(data.keys())
    # 將欲訓練特徵改成正規化
    for category in ['train', 'val', 'test']:

        if range_type == "":
          key = category
        else:
          key = category + "_" + str(feature_id)

        data['x_' + key][..., 0] = data['scaler'].transform(data['x_' + key][..., 0])
        print("data['x_' + key]:", 'x_' + key)

    feature_id += 1

#print(data['x_train'].shape)
data['train_loader'] = DataLoaderM(
    data['x_train'], data['y_train'],
    data['x_train_1'],
    data['x_train_2'],
    data['x_train_3'],
    data['x_train_4'],
    batch_size)

data['val_loader'] = DataLoaderM(
    data['x_val'], data['y_val'],
    data['x_val_1'],
    data['x_val_2'],
    data['x_val_3'],
    data['x_val_4'],
    valid_batch_size)

data['test_loader'] = DataLoaderM(
    data['x_test'], data['y_test'],
    data['x_test_1'],
    data['x_test_2'],
    data['x_test_3'],
    data['x_test_4'],
    test_batch_size)

sensor_ids, sensor_id_to_ind, adj_mx = load_adj(args.adj_data,args.adjtype)   # adjtype: default='doubletransition'

adj_mx = [torch.tensor(i).to(device) for i in adj_mx]

dataloader = data.copy()

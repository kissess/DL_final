def reset_weights(m): #model을 받는다
    ###어떤 모델을 만들고 사용시 parameter 초기화용 함수
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children(): #model의 모든 layer에 있는 parameter들을 for문으로 돈다
    if hasattr(layer, 'reset_parameters'): #이때 'reset_parameters'라는 attribute가 있다면 
        # print(f'Reset trainable parameters of layer = {layer}')
        layer.reset_parameters() #layer에 있는 모든 파라메터 초기화

epoch = 10
sm = nn.Softmax(dim=1)
criterion = nn.CrossEntropyLoss()

fold_train_losses = []
fold_val_losses = []

running_loss = torch.zeros(epoch)
val_loss = torch.zeros(epoch)
accuracy = torch.zeros(epoch)

train_losses = []
val_losses = []


#kf = KFold(n_splits=5, shuffle=True)
for fold, (train_ind, valid_ind) in enumerate(kf.split(train_data)): #enumerate를 통해 fold에 count 반환 
    #kf: instance of KFold() #kf.split(train_data) -> train set과 val set을 튜플 형태로 돌려준다. #4/5는 train set, 즉, 랜덤하게 뽑은 인덱스를 리스트로 돌려준다
    print('Starting fold = ', fold)
    #sampler 정의
    train_sampler_kfold = SubsetRandomSampler(train_ind) #train_ind를 data loader의 sampler 인자가 받을 수 있는 객체로 변환해준다
    valid_sampler_kfold = SubsetRandomSampler(valid_ind)
    
    #data loader 정의
    train_loader_kfold = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler_kfold) 
    # train_data중에서 sampler(train_ind) 인덱스에 해당하는 data를 random으로 batch_size만큼 가져온다 
    valid_loader_kfold = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler_kfold)
    
    model.apply(reset_weights) #파라메터 초기화 -> 새로운 fold에서 즉, 새로운 training, val set에 대해서도 성능을 확인하려면 network을 다시 초기화 하고 그 loss를 저장해야한다.
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    for e in np.arange(epoch):  #(첫번째 fold-첫번째 epoch loss, 첫번째 fold-두번째 epoch loss를 저장 ...)->모든 fold의 모든 epoch별로 training loss가 전부 저장된다.
        print('Starting epoch = ', e)
        
        # Training
        model.train()
        for Xtrain, ytrain in train_loader_kfold:
            # Xtrain = Xtrain.reshape(Xtrain.shape[0],-1)
            optimizer.zero_grad()
            logits = model(Xtrain)
            loss = criterion(logits, ytrain)
            loss.backward()
            optimizer.step()
            
            running_loss[e] += loss.item()
            
        # Validation
        model.eval()
        with torch.no_grad():
            for Xtrain, ytrain in valid_loader_kfold:
                # Xtrain = Xtrian.reshape(Xtrian.shape[0],-1)
                logits = model(Xtrain)   
                val_loss[e] += criterion(logits, ytrain)
                
                ps = sm(logits) 
                top_p, top_class = ps.topk(1,dim=1)
                equals = top_class == ytrain.reshape(top_class.shape)
                accuracy[e] += torch.mean(equals.type(torch.float)) #이미 epoch만큼 0으로 채워진 "torch array"기 때문에(즉, 파이썬 리스트와 달리 크기가 정해져있음) +=로 추가
                
            
        train_losses.append(running_loss/len(train_loader_kfold)) #training loss 모아둔 것/batch size (epoch한번 진행할 때마다 출력) #list니까 append 사용해서 추가
        val_losses.append(val_loss/len(valid_loader_kfold)) #valid_loss 모아둔 것/batch size
        
        print("Epoch: {}/{}.. ".format(e+1, epoch), 
          "Training Loss: {:.3f}.. ".format(running_loss[e]/len(train_loader_kfold)), #현재 epoch에서의 loss(오차들의 평균)출력
          "Validation Loss: {:.3f}.. ".format(val_loss[e]/len(valid_loader_kfold)),
          "Validation Accuracy: {:.3f}".format(accuracy[e]/len(valid_loader_kfold)))
    
# fold_train_losses.append()
# fold_val_losses.append()

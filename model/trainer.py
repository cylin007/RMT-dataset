class Trainer():
    def __init__(self, model, lrate, wdecay, clip, step_size, seq_out_len, scaler, device, cl=True):
        self.scaler = scaler
        self.model = model
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = masked_mae
        self.clip = clip
        self.step = step_size
        self.iter = 1
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl
  
    def train(self, input, input_1, input_2, input_3, input_4 ,real_val, idx=None):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, input_1, input_2, input_3,input_4, idx=idx)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        if self.iter%self.step==0 and self.task_level<=self.seq_out_len:
            self.task_level +=1
            print("### cl learning\n iter",self.iter,"\niter%step",self.iter%self.step,"\ntask_level",self.task_level)
            print("# predict len:", len(predict[:, :, :, :self.task_level]))

        if self.cl:
            loss = masked_mae(predict[:, :, :, :self.task_level], real[:, :, :, :self.task_level], 0.0)
        else:
            loss = masked_mae(predict, real, 0.0)

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()

        metrics = metric(predict, real) # mae,mape,rmse

        self.iter += 1
        return metrics # mae,mape,rmse

    def eval(self, input, input_1, input_2, input_3,input_4, real_val):
        self.model.eval()
        output = self.model(input, input_1, input_2, input_3,input_4)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        metrics = metric(predict, real) # mae,mape,rmse
        return metrics # mae,mape,rmse

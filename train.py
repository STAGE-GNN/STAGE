# hypergnn + hetegnn

from utils import *
from models import HyperSTGNN
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import recall_score as rec
from sklearn.metrics import precision_score as pre
from sklearn.metrics import f1_score as f1
from sklearn.metrics import roc_auc_score as roc
from sklearn.metrics import classification_report 
from imblearn.metrics import geometric_mean_score as gmean
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import logging
import pickle
import utils

random.seed(42)  
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device=torch.device('cuda')

n_epoch =1000
clip = 0.25
lr = 0.001
weight_decay = 1e-4
eta_min = 0
dropout = 0.4 # 0.25
g, features, dict_node_feats = load_hete_graph()

# positional_embedding_size = 32
# g = utils._add_undirected_graph_positional_embedding(g,32)

fin_seq = np.load('data/listed_comp/financial_seq_normal.npy')  # financial data
fin_seq = torch.from_numpy(fin_seq).to(device)

features = features.to(device)

labels_ttl = np.load('data/risk_label.npy')

num_nodes = g.num_nodes()
labels = torch.tensor(labels_ttl)

input_dim = 57
output_dim = 8
total_company_num = g.num_nodes()
rel_num = 3
com_initial_emb = features

best_acc = 0
best_f1 =0 

criterion = torch.nn.CrossEntropyLoss()

train_data,val_data,test_data = split_data()

# tr_idx = np.load('model_save/train_idx_best.npy')
# val_idx = np.load('model_save/val_idx_best.npy')
# t_idx = np.load('model_save/test_idx_best.npy')
# train_data, val_data, test_data = my_split_data(tr_idx, val_idx, t_idx)

train_idx = train_data.indices
valid_idx = val_data.indices
test_idx = test_data.indices

# np.save('model_save/train_idx1.npy',train_idx)
# np.save('model_save/val_idx1.npy',valid_idx)
# np.save('model_save/test_idx1.npy',test_idx)

# train_data,val_data,test_data = my_load_data() 
# train_idx = train_data.indices.tolist()
# valid_idx = val_data.indices.tolist()
# test_idx = test_data.indices.tolist()

train_hyp_graph = load_sub_hyper_graph(train_data)
val_hyp_graph = load_sub_hyper_graph(val_data)
test_hyp_graph = load_sub_hyper_graph(test_data)

def train():

    st_time = time.time()

    global best_acc, best_f1
    gnn = HyperSTGNN(input_dim,output_dim,
                     total_company_num,rel_num,
                     device,com_initial_emb,g,dict_node_feats,
                     fin_seq=fin_seq,
                     num_heads=1,dropout=dropout,norm=True).to(device)

    classifier = Classifier(output_dim, 2).to(device)
    # classifier = Classifier(output_dim, 2)
    model = nn.Sequential(gnn, classifier).to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=eta_min)

    ['industry', 'area', 'qualify']
    train_hyp=[]
    for i in ['industry']:
        train_hyp+=[gen_attribute_hg(total_company_num, train_hyp_graph[i], X=None)]
    valid_hyp=[]
    for i in ['industry']:
        valid_hyp+=[gen_attribute_hg(total_company_num, val_hyp_graph[i], X=None)]
    test_hyp=[]
    for i in ['industry']:
        test_hyp+=[gen_attribute_hg(total_company_num, test_hyp_graph[i], X=None)]


    for epoch in np.arange(n_epoch):
        # for batch in range(iters):
        st=time.time()
        '''
            Train 
        '''
        model.train()
        train_losses = []
        # torch.cuda.empty_cache()

        company_emb=gnn.forward(g,dict_node_feats,train_hyp,train_idx)

        res = classifier.forward(company_emb)
        train_label = labels[train_idx]
        loss = criterion(res.cpu(), torch.LongTensor(train_label).cpu())
        optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        #     loss.backward()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        train_losses += [loss.cpu().detach().tolist()]
            # train_step += 1
        scheduler.step()
        del res, loss

        '''
            Valid 
        '''
        model.eval()
        with torch.no_grad():
            company_emb=gnn.forward(g,dict_node_feats,valid_hyp,valid_idx)

            res = classifier.forward(company_emb).cpu()
            valid_label = labels[valid_idx]
            loss = criterion(res.cpu(),torch.LongTensor(valid_label) )

            pred=res.argmax(dim=1).cpu()
            ac=acc(valid_label,pred)
            pr=pre(valid_label,pred)
            re=rec(valid_label,pred)
            f=f1(valid_label,pred)
            rc=roc(valid_label,res[:,1])
            gm = gmean(valid_label,pred)
            fpr, tpr, thresholds = roc_curve(valid_label, pred)
            ac = auc(fpr, tpr)

            if ac > best_acc and f>best_f1:
                best_acc = ac
                best_f1=f
                torch.save(model, './model_save/%s.pkl'%('best_model'))

            et = time.time()
            print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid Acc: %.4f Valid Pre: %.4f  Valid Recall: %.4f Valid F1: %.4f  Valid Roc: %.4f Valid Gmean: %.4f Valid AUC: %.4f"  ) % \
                (epoch, (et - st), optimizer.param_groups[0]['lr'], np.average(train_losses), \
                loss.cpu().detach().tolist(), ac,pr,re,f,rc,gm,ac))

            # print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid Acc: %.4f Valid Pre: %.4f  Valid Recall: %.4f Valid F1: %.4f"  ) % \
            #     (epoch, (et - st), optimizer.param_groups[0]['lr'], np.average(train_losses), \
            #     loss.cpu().detach().tolist(), ac,pr,re,f))
            
            del res, loss

            if epoch+1==n_epoch:
                company_emb=gnn.forward(g,dict_node_feats,test_hyp,test_idx)
                # gnn.forward(g,dict_node_feats,valid_hyp,valid_idx)
                test_label = labels[test_idx]
                res = classifier.forward(company_emb).cpu()

                pred=res.argmax(dim=1).cpu()
                ac=acc(test_label,pred)
                pr=pre(test_label,pred)
                re=rec(test_label,pred)
                f=f1(test_label,pred)
                rc=roc(test_label,res[:,1])
                gm = gmean(test_label,pred)
                fpr, tpr, thresholds = roc_curve(test_label, pred)
                ac = auc(fpr, tpr)
                
                #print optimizer's state_dict
                print('############################################################')
                print('Optimizer,s state_dict:')
                for var_name in ['param_groups']: # ['param_groups'] / optimizer.state_dict()
                    print(var_name,'\t',optimizer.state_dict()[var_name])

                print('############################################################')
                print('Last Test Acc: %.4f Last Test Pre: %.4f Last Test Recall: %.4f Last Test F1: %.4f Last Test ROC: %.4f Last Test Gmean: %.4f Last Test AUC: %.4f' % (ac,pr,re,f,rc, gm, ac))
                # print('Last Test Acc: %.4f Last Test Pre: %.4f Last Test Recall: %.4f Last Test F1: %.4f' % (ac,pr,re,f))
                # print(classification_report(pred,test_label))
                end_time = time.time()
                print(f"Training time: {end_time-st_time} seconds")

train()
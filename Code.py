import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tkinter import *
from tkinter import messagebox
from tkinter import ttk


def data_encoder_KQ(X):
    for i, j in enumerate(X):
        for k in range(0, 5):
            if(j[k] =="Nu"):
                j[k]=2
            elif(j[k] =="Nam"):
                j[k] = 3
            elif(j[k] == "A"):
                j[k]=4
            elif(j[k] == "B"):
                j[k]=5
            elif(j[k] == "C"):
                j[k]=6
            elif(j[k] == "D"):
                j[k]=7
            elif(j[k] == "E"):
                j[k]=8
            elif(j[k] == "Cu nhan DH"):
                j[k]=9
            elif(j[k] == "Nhieu truong DH"):
                j[k]=10
            elif(j[k] == "Thac sy"):
                j[k]=11
            elif(j[k] == "Cao dang"):
                j[k]=12
            elif(j[k] == "THPT"):
                j[k]=13
            elif(j[k] == "GDTX"):
                j[k]=14
            elif(j[k] == "Tieu chuan"):
                j[k]=15
            elif(j[k] == "Khong"):
                j[k]=16
            elif(j[k] == "Chua"):
                j[k]=17
            elif(j[k] == "Hoan thanh"):
                j[k]=18
            elif(j[k] == "Yes"):
                j[k]=1
            elif(j[k] == "No"):
                j[k]=0
    return X
    


def data_encoder(X):
    for i, j in enumerate(X):
        for k in range(0, 6):
            if(j[k] =="Nu"):
                j[k]=2
            elif(j[k] =="Nam"):
                j[k] = 3
            elif(j[k] == "A"):
                j[k]=4
            elif(j[k] == "B"):
                j[k]=5
            elif(j[k] == "C"):
                j[k]=6
            elif(j[k] == "D"):
                j[k]=7
            elif(j[k] == "E"):
                j[k]=8
            elif(j[k] == "Cu nhan DH"):
                j[k]=9
            elif(j[k] == "Nhieu truong DH"):
                j[k]=10
            elif(j[k] == "Thac sy"):
                j[k]=11
            elif(j[k] == "Cao dang"):
                j[k]=12
            elif(j[k] == "THPT"):
                j[k]=13
            elif(j[k] == "GDTX"):
                j[k]=14
            elif(j[k] == "Tieu chuan"):
                j[k]=15
            elif(j[k] == "Khong"):
                j[k]=16
            elif(j[k] == "Chua"):
                j[k]=17
            elif(j[k] == "Hoan thanh"):
                j[k]=18
            elif(j[k] == "Yes"):
                j[k]=1
            elif(j[k] == "No"):
                j[k]=0
    return X
df = pd.read_csv('examschuan.csv')
X_data = np.array(df[['GioiTinh', 'NhomHoc', 'TrinhDoHocVanBaMe', 'CheDoAnUong','KhoaLuyenThi','TrangThai']].values)    
X=data_encoder(X_data)
X=X.astype('int32')

dt_Train, dt_Test = train_test_split(X,test_size=0.3, shuffle=False)
x_test_dt_test = dt_Test[:, :5]
y_test_dt_test = dt_Test[:,5]
#print(rate_svc)
k=5
kf = KFold(n_splits=k,random_state=None)

def error(y, y_pred):
    l = []
    for i in range(len(y)):
        l.append(np.abs(np.array(y[i:i+1]) - np.array(y_pred[i:i+1])))
    return np.mean(l)
max_svc =999999
i=1

for train_index,test_index in kf.split(dt_Train): 
    X_train = dt_Train[train_index,:5]
    X_test = dt_Train[test_index,:5]
    Y_train = dt_Train[train_index,5]
    Y_test = dt_Train[test_index,5]
    svc_kfold = SVC(kernel='linear')
    svc_kfold.fit(X_train, Y_train)
    y_pred_train = svc_kfold.predict(X_train)
    y_pred_test = svc_kfold.predict(X_test)

    sum = error(Y_train, y_pred_train) + error(Y_test, y_pred_test)
    if sum < max_svc:
        max_svc = sum
        last = i
        model_max = svc_kfold.fit(X_train, Y_train)
    i += 1
y_predict = model_max.predict(x_test_dt_test)
rate_svc_kfold = round(accuracy_score(y_predict, y_test_dt_test),2)
pre_svc_kfold = np.round(precision_score(y_test_dt_test, y_predict, average='micro', zero_division=1),3)
recall_svc_kfold = np.round(recall_score(y_test_dt_test, y_predict, average='micro', zero_division=1),3)
f1_kfold = np.round(f1_score(y_test_dt_test, y_predict, average='macro', zero_division=1),3)

form = Tk()
form.title("Dự đoán khả năng đỗ kì thi của học sinh:")
form.geometry("700x300")

lable_GioiTinh = Label(form, text = "Nhập giới tính học sinh:")
lable_GioiTinh.grid(row = 1, column = 1, padx = 30, pady = 10)
textbox_GioiTinh = Entry(form)
textbox_GioiTinh.grid(row = 1, column = 2)

lable_NhomHoc = Label(form, text = "Nhập nhóm học:")
lable_NhomHoc.grid(row = 2, column = 1, padx = 10, pady = 10)
textbox_NhomHoc = Entry(form)
textbox_NhomHoc.grid(row = 2, column = 2)

lable_TrinhDoHocVan = Label(form, text = "Nhập trình độ học vấn ba mẹ:")
lable_TrinhDoHocVan.grid(row = 3, column = 1,padx = 30, pady = 10)
textbox_TrinhDoHocVan = Entry(form)
textbox_TrinhDoHocVan.grid(row = 3, column = 2)

lable_CheDoAnUong = Label(form, text = "Nhập chế độ ăn uống:")
lable_CheDoAnUong.grid(row = 4, column = 1,padx = 30,pady = 10)
textbox_CheDoAnUong = Entry(form)
textbox_CheDoAnUong.grid(row = 4, column = 2)

lable_LuyenThi = Label(form, text = "Nhập khóa luyện thi:")
lable_LuyenThi.grid(row = 5, column = 1, padx = 30,pady = 10)
textbox_LuyenThi = Entry(form)
textbox_LuyenThi.grid(row = 5, column = 2)


lb_svc = Label(form)
lb_svc.grid(column=3, row=3, padx = 70)
lb_svc.configure(text=""+'\n' 
                           +"Tỉ lệ dự đoán đúng: "+str("...")+"%"+'\n'
                           +"Precision: "+str("...")+'\n'
                           +"Recall: "+str("...")+'\n'
                           +"F1-score: "+str("..."), padx=20)
lbl3 = Label(form, text="...")
lbl3.grid(column=3, row=2)
def dudoansvm():
    GioiTinh = textbox_GioiTinh.get()
    NhomHoc = textbox_NhomHoc.get()
    TrinhDoHocVan = textbox_TrinhDoHocVan.get()
    CheDoAnUong = textbox_CheDoAnUong.get()
    LuyenThi =textbox_LuyenThi.get()
   
    if((GioiTinh == '') or (NhomHoc == '') or (TrinhDoHocVan == '') or (CheDoAnUong == '') or (LuyenThi == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
            
            X_dudoan =np.array([GioiTinh,NhomHoc,TrinhDoHocVan,CheDoAnUong,LuyenThi]).reshape(1, -1)
            X_encode = data_encoder_KQ(X_dudoan)
            y_kqua = model_max.predict(X_encode)
            if(y_kqua==1):
                lbl3.configure(text= "[Yes]")
                lb_svc.configure(text=""+'\n' 
                           +"Tỉ lệ dự đoán đúng: "+str(rate_svc_kfold*100)+"%"+'\n'
                           +"Precision: "+str(pre_svc_kfold)+'\n'
                           +"Recall: "+str(recall_svc_kfold)+'\n'
                           +"F1-score: "+str(f1_kfold), padx=20)
            else:
                lbl3.configure(text= "[No]")
                
##    else:
##           messagebox.showinfo("Thông báo", "Bạn Nhập sai thông tin yêu cầu!")
def reset():
    textbox_GioiTinh.delete(0,END)
    textbox_NhomHoc.delete(0,END)
    textbox_TrinhDoHocVan.delete(0,END)
    textbox_CheDoAnUong.delete(0,END)
    textbox_LuyenThi.delete(0,END)
    lbl3.configure(text= "...")
    lb_svc.configure(text=""+'\n' 
                           +"Tỉ lệ dự đoán đúng: "+str("...")+"%"+'\n'
                           +"Precision: "+str("...")+'\n'
                           +"Recall: "+str("...")+'\n'
                           +"F1-score: "+str("..."), padx=20)
button_svc = Button(form, text = 'Kết quả dự đoán theo SVM', command = dudoansvm)
button_svc.grid(row = 1, column = 3, pady = 20)

button_svc = Button(form, text = 'Reset', command = reset)
button_svc.grid(row = 5, column = 3, pady = 20)
form.mainloop()



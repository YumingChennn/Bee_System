import os
import pandas as pd
import pymongo
import gridfs

# Step 1: 讀取 Excel 資料
excel_file = 'Fake_Dataset_New.xlsx'  # 替換成你的 Excel 檔案路徑
df = pd.read_excel(excel_file)

# Step 2: 依照順序排列資料
df_sorted = df.sort_values(by='Date')  # 假設 'Data' 是日期欄位名稱

# Step 3: 連接 MongoDB
client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['audio_db']  # 創建一個資料庫，名稱是 'combined_db'
collection = db['caudio_db']  # 創建一個集合來存儲資料
fs = gridfs.GridFS(db)  # 使用 GridFS 處理音訊檔案

# Step 4: 設定聲音檔資料夾路徑
folder_path = 'sound_files/Mydataset/Datasettest'

# Step 5: 將 Excel 資料與聲音檔案進行配對並上傳到 MongoDB
for i, (index, row) in enumerate(df_sorted.iterrows()):
    # 確保資料順序一致
    filename = os.listdir(folder_path)[i]  # 根據順序取得對應的聲音檔案名稱
    if filename.endswith('.wav'):
        file_path = os.path.join(folder_path, filename)

        # 讀取音訊檔案
        with open(file_path, 'rb') as f:
            audio_data = f.read()

        # 將音訊檔案上傳至 GridFS 並取得音訊檔案 ID
        audio_id = fs.put(audio_data, filename=filename)

        # 將 Excel 資料和音訊檔案 ID 組合
        document = {
            'Date': row['Date'],
            'Box Number': row['Box Number'],
            'Weight': row['Weight'],
            'Temperature': row['Temperature'],
            'Humidity': row['Humidity'],
            'audio_file_id': audio_id  # 存儲音訊檔案的 GridFS ID
        }
        # 將資料插入到 MongoDB
        collection.insert_one(document)

print("所有資料和對應的音檔已成功上傳至 MongoDB。")

import gdown

#Download Data from google drive url

def download_data():
    train_url = 'https://drive.google.com/uc?id=1QV7r1Gr6Qh8lB-cV5Zui5_2ElQoQgYbb'
    valid_url = 'https://drive.google.com/uc?id=1MmDF2k4s7VrlWRqyOtw-KG5pHF9P7u9v'
    valid4test_url = 'https://drive.google.com/uc?id=1T5UFbIWq8IA5ox0upGcpxtTRyJwakxwI'

    gdown.download(train_url, './data/train.json')
    gdown.download(valid_url, './data/valid.json')
    gdown.download(valid4test_url, './data/test.json')

if __name__ == "__main__":
    download_data()
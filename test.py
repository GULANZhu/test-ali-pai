import datetime
# import oss2

time=datetime.datetime.now()
time = datetime.datetime.strftime(time,'%Y-%m-%d %H-%M-%S')
a=1
a=2
a=3
a=4
print(time)

# sdfsdfs



# auth = oss2.Auth('LTAI4G5FSAqJg5Wyham9uTXN', 'zj0ejgnEyLUutaPe8Zg4Ra1hVqZUZS')
# bucket = oss2.Bucket(auth, 'http://oss-cn-beijing-internal.aliyuncs.com', 'test-experiment-0202-2')

# #读取一个完整文件。
# result = bucket.get_object('/test_torch/1/test.pt')
# print(result.read())
# #按Range读取数据。
# result = bucket.get_object('<your_file_path/your_file>', byte_range=(0, 99))
# #写数据至OSS。
# bucket.put_object('<your_file_path/your_file>', '<your_object_content>')
# #对文件进行Append。
# result = bucket.append_object('<your_file_path/your_file>', 0, '<your_object_content>')
# result = bucket.append_object('<your_file_path/your_file>', result.next_position, '<your_object_content>')

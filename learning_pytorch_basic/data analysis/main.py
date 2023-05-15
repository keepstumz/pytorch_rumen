from typing import List

from data_define import Record
from file_define import File_reader, TextFile_reader, JsonFile_reader


text_file_reader = TextFile_reader("D:/textfile.txt")
Json_file_reader = JsonFile_reader("D:/jsonfile.txt")

"""将一月份和二月份的数据导出"""
jan_data: List[Record] = text_file_reader.read_data()
feb_data: List[Record] = Json_file_reader.read_data()

"""将两个月份的数据整合在一起"""
all_data: List[Record] = jan_data + feb_data

"""建立一个数据字典，key值为日期，value为当日的营销额"""
data_list = {}
for record in all_data:
    if record.date in data_list:
        data_list[record.date] += record.money
    else:
        data_list[record.date] = record.money
print(data_list)

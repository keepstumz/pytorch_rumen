"""构造一个抽象类，定义文件读取的相关功能，并使用子类及实现具体功能，针对不同类型的数据，所以采取不同的子类，这里是继承和多态的实例展现"""
import json

from typing import List

from data_define import Record


class File_reader:
    """初始构造函数，将文件的地址在创建该类时就赋值进去"""

    def __init__(self, path):
        self.path = path

    def read_data(self) -> List[Record]:
        pass


class TextFile_reader(File_reader):
    """构造读取数据的成员方法，将文件内每一行的数据读取出来然后进行分割处理输入到list类型中封存"""
    """父类中方法的复写，多态的展现"""
    def read_data(self):
        f = open(f"{self.path}", "r", encoding="UTF-8")
        data_list: List[Record] = []  # 最后的数据列表是装入之前定义好的数据类的，所以其中每一个数据都是Record类型
        for line in f.readlines():
            line = line.strip()  # 去除每一行的最后换行符
            data: List[str] = line.split(",")
            # print(data)
            record = Record(data[0], data[1], int(data[2]), data[3])
            data_list.append(record)
        f.close()
        return data_list


class JsonFile_reader(File_reader):
    """构造读取数据的成员方法，将文件内每一行的数据读取出来然后进行分割处理输入到list类型中封存"""

    def read_data(self):
        f = open(f"{self.path}", "r", encoding="UTF-8")
        data_list: List[Record] = []
        for line in f.readlines():
            data_dict = json.loads(line)
            # print(data_dict)
            record: Record = Record(data_dict['data'], data_dict['order_id'], int(data_dict['money']), data_dict['province'])
            data_list.append(record)
        return data_list


if __name__ == "__main__":
    text_file_reader = TextFile_reader("D:/textfile.txt")
    list1 = text_file_reader.read_data()
    for i in list1:
        print(i)

    Json_file_reader = JsonFile_reader("D:/jsonfile.txt")
    list2 = Json_file_reader.read_data()
    for i in list2:
        print(i)

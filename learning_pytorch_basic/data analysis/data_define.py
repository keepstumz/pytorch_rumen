"""设计一个类，可以完成数据的封装，对于一条一条的数据封装进list便于调用"""


class Record:
    """类的构造函数，通过init构造生成，生成时传入类的成员"""

    def __init__(self, date, order_id, money, province):
        self.date = date
        self.order_id = order_id
        self.money = money
        self.province = province

    def __str__(self):
        return f"{self.date}, {self.order_id}, {self.money}, {self.province}"

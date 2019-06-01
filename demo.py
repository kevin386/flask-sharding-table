# coding: utf8
from flask import Flask
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///demo.db'

db = SQLAlchemy(app)


class ShardingModel(db.Model):
    # 最大的分表数量,过多的分表不好,限制下最大值
    _max_table_size = 100
    # 缓存class对象
    _cls_mapper = {}
    # 分表数量,可覆写,最大1000个
    _sharding_num = 10

    # 继承ShardingModel的model也必须是抽象类
    __abstract__ = True

    def __repr__(self):
        """
        主键都是id，默认规则不要变
        :return:
        """
        return "{}: <table name: {}>".format(self.__class__.__name__, self.__tablename__)

    @classmethod
    def str_camel_to_underline(cls, s):
        """
        把驼峰命名的字符串转换为小写+下划线
        :param s:
        :return:
        """
        if not isinstance(s, str):
            return s
        return ''.join([c if c.islower() else '_' + c.lower() for c in s.strip()]).strip('_')

    @classmethod
    def init_sharding_table(cls):
        """
        在db.create_all()之前调用，让db知道新建了表
        :return:
        """
        for idx in range(cls._sharding_num):
            cls.model(idx)

        # for cls_name, cls_obj in sorted(cls._cls_mapper.items(), key=lambda x: x[0]):
        #     print 'init sharding table, class name: {}, table name: {}'.format(cls_name, cls_obj.__tablename__)

    @classmethod
    def get_table_idx(cls, value):
        """
        用model中的某一列的值来分表，比如user id
        :param value:
        :return:
        """
        return int(value) % cls._sharding_num

    @classmethod
    def model(cls, value):
        """
        生成类对象
        :param value:
        :return:
        """
        table_index = cls.get_table_idx(value)
        if table_index >= cls._max_table_size:
            return

        table_name = '%s_%0d' % (cls.str_camel_to_underline(cls.__name__), table_index)
        class_name = '%s%0d' % (cls.__name__, table_index)

        ModelClass = cls._cls_mapper.get(class_name, None)
        if ModelClass is None:
            ModelClass = type(class_name, (cls,), {
                '__module__': __name__,
                '__name__': class_name,
                '__tablename__': table_name,
            })
            cls._cls_mapper[class_name] = ModelClass

        return ModelClass


class GlobalUserID(db.Model):
    """
    全局用户ID自增，确保分表后的用户id不重复
    """
    id = db.Column(db.Integer, primary_key=True)

    @staticmethod
    def make_user_id():
        user_id = GlobalUserID()
        db.session.add(user_id)
        db.session.commit()
        return user_id.id

    @classmethod
    def get_max_user_id(cls):
        return cls.query.order_by(db.desc('id')).first()


# 在表创建完成之后，设置自增起始值，在Colum设置设置autoincrement是无效的
# sqlite3中会报错，导致建表失败，先注释掉
# from sqlalchemy import event
# from sqlalchemy import DDL
# event.listen(
#     GlobalUserID.__table__,
#     "after_create",
#     DDL("ALTER TABLE %(table)s AUTOINCREMENT = {};".format(GlobalUserID.AUTOINCREMENT_INIT))
# )


class User(ShardingModel):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    email = db.Column(db.String(100))

    # 继承ShardingModel必须使用抽象类，不要实例化
    __abstract__ = True

    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

    def __repr__(self):
        """
        主键都是id，默认规则不要变
        :return:
        """
        return "{}: <id: {}, username: {}, email: {}>".format(
            self.__class__.__name__, self.id, self.username, self.email
        )

    @classmethod
    def create_user(cls, username, email):
        """
        创建用户
        :param username:
        :param email:
        :return:
        """
        user_id = GlobalUserID.make_user_id()
        user = cls.model(user_id)(user_id, username, email)
        db.session.add(user)
        db.session.commit()
        return user

    @classmethod
    def get_user(cls, user_id):
        """
        查询用户
        :param user_id:
        :return:
        """
        return cls.model(user_id).query.get(user_id)


User.init_sharding_table()


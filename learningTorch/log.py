import logging

# 简单操作
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    filename='./test.log',
                    level=logging.DEBUG,    # level控制打印级别
                    filemode='a')           # filename输出的文件名，filemode，'a'为追加模式,'w'为覆盖模式
'''
%(name)s：Logger的名字
%(levelno)s：打印日志级别的数值
%(levelname)s：打印日志级别的名称
%(pathname)s：打印当前执行程序的路径，其实就是sys.argv[0]
%(filename)s：打印当前执行程序名
%(funcName)s：打印日志的当前函数
%(lineno)d：打印日志的当前行号
%(asctime)s：打印日志的时间
%(thread)d：打印线程ID
%(threadName)s：打印线程名称
%(process)d：打印进程ID
%(message)s：打印日志信息
'''
s = '123'
logging.debug('debug级别，一般用来打印一些调试信息，级别最低%s', s)
logging.info('Epoch: [{0}][{1}/{2}]\t'.format(1, 2, 3))
logging.warning('waring级别，一般用来打印警告信息%s', s)
logging.error('error级别，一般用来打印一些错误信息%s', s)
logging.critical('critical级别，一般用来打印一些致命的错误信息，等级最高')


# 模块化操作
logger = logging.getLogger('test')
logger.setLevel(level=logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelno)s: %(message)s')

file_handler = logging.FileHandler('./test2.log')
file_handler.setLevel(level=logging.INFO)
file_handler.setFormatter(formatter)


stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

s = '123'
logger.debug('debug级别，一般用来打印一些调试信息，级别最低%s', s)
logger.info('Epoch: [{0}][{1}/{2}]\t'.format(1, 2, 3))
logger.warning('waring级别，一般用来打印警告信息%s', s)
logger.error('error级别，一般用来打印一些错误信息%s', s)
logger.critical('critical级别，一般用来打印一些致命的错误信息，等级最高')


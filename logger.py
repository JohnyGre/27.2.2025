import logging

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log(message, level='info'):
    if level == 'error':
        logging.error(message)
    else:
        logging.info(message)


def debug(param):
    return None
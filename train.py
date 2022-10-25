from lib.core.utils.logger import logger
from lib.core.base_trainer.network import Train
import setproctitle

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

logger.info('train start')
setproctitle.setproctitle("detect")

trainner=Train()




trainner.custom_loop()

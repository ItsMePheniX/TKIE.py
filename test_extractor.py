import logging
logging.basicConfig(level=logging.DEBUG)

from tkie.extractor import TKIEExtractor

extractor = TKIEExtractor()
extractor.process("test.png")

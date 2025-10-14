import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/rl2-bonjour/EgoVerse_New/EgoVerse/Robot/Eva/install/eva'

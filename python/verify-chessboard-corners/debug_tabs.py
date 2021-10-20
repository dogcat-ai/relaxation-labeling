# This class is a debugging mechanism, utilizing good old-fashioned print-outs.
#   The specifics of the print-out are left to the user of this class.
#   The main functionality that this class provides is accounting for nested debug statements.
#   Inner statements receive one additional tab relative to their parent.
#   This additional tab is always applied when entering a new function,
#   and is always taken away upon exiting said new function.
#   Nonetheless, the user can increment the number of tabs within a function
#   (and likewise decrement).
#   This will often be done around/within a for loop.

class DebugTabs:
    def __init__(self):
        self.init()

    def init(self):
        self.tabs = ''
        self.len = 0

    def reset(self):
        self.init()

    def increment(self):
        self.tabs += '\t'
        self.len += 1

    def decrement(self):
        if (self.len > 0):
            self.tabs = self.tabs[1:]
            self.len -= 1

    def receive(self, debugTabsLen):
        self.len = debugTabsLen
        self.tabs = ''
        for i in range(self.len):
            self.tabs += '\t'

    def print(self, statement, end='\n', sep=' ', preincrement=0, postincrement=0):
        if (preincrement > 0):
            for i in range(preincrement):
                self.increment()
        elif (preincrement < 0):
            for i in range(preincrement,preincrement-1,-1):
                self.decrement()
        print('{}{}'.format(self.tabs, statement), end=end, sep=sep)
        if (postincrement > 0):
            for i in range(postincrement):
                self.increment()
        elif (postincrement < 0):
            for i in range(postincrement,postincrement-1,-1):
                self.decrement()

def __main__():
    # Testing:
    #"""
    debugTabs = DebugTabs()
    debugTabs.increment()
    debugTabs.print('test print')
    debugTabs.decrement()
    debugTabs.print('test print')
#"""

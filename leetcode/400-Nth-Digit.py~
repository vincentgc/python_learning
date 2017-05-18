##自己写的
##
def findNthDigit(n):
        """
        :type n: int
        :rtype: int
        """
	#不断减位数，比如前9个减9*1，之后减9*2*10
        m=0
        while n>0:
            new_n = n
            n = n-9*(m+1)*(10**m)
            m=m+1
	#确定是哪个数
        count = (new_n-1)/m+1
        num = 10**(m-1)+(count-1)
	##再看是这个数的哪一位
        i=(new_n-1)%m
        temp=num/(10**(m-1-i))
        result=temp%10
        return result
##网上答案（比较流氓，用str变成字符直接判断是哪一位，不是纯数学办法）
"""
def findNthDigit(self, n):
        start, size, step = 1, 1, 9
        while n > size * step:
            n, size, step, start = n - (size * step), size + 1, step * 10, start * 10
        return int(str(start + (n - 1) // size)[(n - 1) % size])
"""
"""
def findNthDigit(self, n):
        start, size = 1, 1
        while n > size:
            n, start = n - size, start + 1
            size = len(str(start))
        return int(str(start)[n-1])
"""

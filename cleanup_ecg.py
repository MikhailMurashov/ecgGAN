import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


arr = np.load('../data/fix_signals_400.npy')

print(arr.shape)
fixed = []
# lead = 4
plt.subplots_adjust(bottom=0.2)
l, = plt.plot(arr[0])

but_len = Button(plt.axes([0.2, 0.9, 0.2, 0.075]), 'len 0')
but_left = Button(plt.axes([0.6, 0.9, 0.2, 0.075]), 'left '+str(arr.shape[0]))


class Index(object):
    ind = 0

    def ok(self, event):
        fixed.append(arr[self.ind])
        self.ind += 1
        but_len.label.set_text('len ' + str(len(fixed)))
        but_left.label.set_text('left ' + str(arr.shape[0]-self.ind))
        l.set_ydata(arr[self.ind])
        plt.draw()

    def not_ok(self, event):
        self.ind += 1
        but_len.label.set_text('len ' + str(len(fixed)))
        but_left.label.set_text('left ' + str(arr.shape[0]-self.ind))
        l.set_ydata(arr[self.ind])
        plt.draw()

    def prev(self, event):
        fixed.pop()
        self.ind -= 1
        but_len.label.set_text('len ' + str(len(fixed)))
        but_left.label.set_text('left ' + str(arr.shape[0]-self.ind))
        l.set_ydata(arr[self.ind])
        plt.draw()


callback = Index()
but_ok = Button(plt.axes([0.80, 0.05, 0.12, 0.075]), 'OK')
but_not_ok = Button(plt.axes([0.40, 0.05, 0.12, 0.075]), 'NOT OK')
but_prev = Button(plt.axes([0.20, 0.05, 0.12, 0.075]), 'PREV')

but_ok.on_clicked(callback.ok)
but_not_ok.on_clicked(callback.not_ok)
but_prev.on_clicked(callback.prev)

plt.show()

res_arr = np.array(fixed)
print(res_arr.shape)

np.save('../data/fixed/fix_signals_400.npy', res_arr)

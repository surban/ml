import time
import datetime

try:    
    import IPython.core.display
    have_notebook = True
except ImportError:
    have_notebook = False

start_time = 0

def status(current_iter, max_iter, caption=""):
    global start_time

    if current_iter == 0:
        start_time = time.time()
        time_left = "?"
    else:
        cur_time = time.time()
        iters_per_sec = (cur_time - start_time) / current_iter
        secs_left = (max_iter - current_iter) * iters_per_sec
        d = datetime.datetime(1,1,1) + datetime.timedelta(seconds=secs_left)
        time_left = "%d:%02d:%02d" % (d.hour, d.minute, d.second)

    if caption is None:
        return
    if len(caption) > 0:
        desc = "%s: " % caption
    else:
        desc = ""
    if have_notebook:
        IPython.core.display.clear_output(stdout=False, stderr=False, other=True)
        IPython.core.display.display_html(
             '<i>%s</i><meter value="%d" min="0" max="%d">%d / %d</meter> %s left' 
             % (desc, current_iter, max_iter, current_iter, max_iter, time_left), raw=True)
    else:
        print "%s%d / %d (%s left)                             \r" \
            % (desc, current_iter, max_iter, time_left)

def done():
    if have_notebook:
        IPython.core.display.clear_output(stdout=False, stderr=False, other=True)
    else:
        print "                                                            \r"





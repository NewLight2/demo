import datetime

fw = open('weather.txt.out', 'w')

with open('weather.txt') as f:
    for line in f:
        cols = line.strip().split(',')
        date = cols[0]
        dat, time, pm = date.split(' ')
        if pm == 'PM':
            t1, t2 = time.split(':')
            if t1 != '12':
                time = str(int(t1) + 12) + ':' + t2
            else:
                time = "00:00"
        date1 = dat + " " + time + ":00"
        date1 = date1.replace('/', '-')
        date2 = datetime.datetime.strftime(datetime.datetime.strptime(date1, '%Y-%m-%d %H:%M:%S'), '%Y%m%d%H')
        outline = date2 + ',' + ','.join(cols[1:]) + '\n'
        fw.write(outline)


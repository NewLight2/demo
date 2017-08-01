$("#import").click(function () {//点击导入按钮，使files触发点击事件，然后完成读取文件的操作。
    $("#file6").click();
});


var progress = document.getElementById('progress');
var input6 = document.getElementById('file6');
var block = 1 * 1024 * 1024; // 每次读取1M
// 当前文件对象
var file;
// 当前已读取大小
var fileLoaded;
// 文件总大小
var fileSize;

var startStr;
var endStr;

// 每次读取一个block
function readBlob() {

    //var r = document.getElementById('range');

    var blob;
    if (file.webkitSlice) {
        blob = file.webkitSlice(fileLoaded, fileLoaded + block + 1);
    } else if (file.mozSlice) {
        blob = file.mozSlice(fileLoaded, fileLoaded + block + 1);
    } else if (file.slice) {
        blob = file.slice(fileLoaded, fileLoaded + block + 1);
    } else {
        alert('不支持分段读取！');
        return false;
    }
    //reader.readAsBinaryString(blob);
    reader.readAsText(blob);
}

//var n = 0;

// 每个blob读取完毕时调用
function drawPoi(e) {
    var res = this.result;
    var lines = res.split('\n');
    for(var i=0; i<lines.length; i++) {
        var line = lines[i].replace(/\"/g, "");

        if(i == 0) {
            if(startStr) {
               line = startStr + line;
            }
            startStr = null;
        }
        if(i == lines.length-1) {
            startStr = line;
            break;
        }

        var cells = line.split(",");
        if (cells[lng_col] && cells[lat_col] && cells[lng_col]!='' && cells[lat_col]!='') {
            //gps转换成百度经纬度
            var bd09 = toBD09(cells[lng_col] * 1, cells[lat_col] * 1);
            points.push({
                lng: bd09[0],
                lat: bd09[1],
                count: 1
            });
        }
    }

    fileLoaded += e.total;
    var percent = fileLoaded / fileSize;
    progress.style.display = 'block';
    if (percent < 1) {
        // 继续读取下一块
        readBlob();
    } else {
        // 结束
        percent = 1;
        console.log(points.length);
        var layer = new Mapv.Layer({
            mapv: mapv, // 对应的mapv实例
            zIndex: 1, // 图层层级
            context: 'webgl',
            data: points, // 数据
            drawType: 'simple', // 展示形式
            drawOptions: { // 绘制参数
                fillStyle: '#d340c3', // 填充颜色
                size: 3 // 半径
            }
        });
        progress.style.display = 'none';
    }
    percent = Math.ceil(percent * 100) + '%';
    progress.innerHTML = '正在导入数据 ' + percent;
}

function getPoi(e, res, point) {
    var lines = res.split('\n');
    for(var i=0; i<lines.length; i++) {
        var line = lines[i].replace(/\"/g, "");

        if(i == 0) {
            if(startStr) {
                line = startStr + line;
            }
            startStr = null;
        }
        if(i == lines.length-1) {
            startStr = line;
            break;
        }

        var cells = line.split(",");
        if (cells[lng_col] && cells[lat_col] && cells[lng_col]!='' && cells[lat_col]!='') {
            //gps转换成百度经纬度
            var bd09 = toBD09(cells[lng_col] * 1, cells[lat_col] * 1);
            if(bmap.getDistance(point, new BMap.Point(bd09[0], bd09[1])) < 20) {
                cells[cells.length] = find_loc(bd09[0], bd09[1]);
                data.push(cells);
            }
        }
    }

    fileLoaded += e.total;
    var percent = fileLoaded / fileSize;
    progress.style.display = 'block';
    if (percent < 1) {
        // 继续读取下一块
        readBlob();
    } else {
        // 结束
        percent = 1;
        progress.style.display = 'none';
        console.log(data.length);
        showInfo();
    }
    percent = Math.ceil(percent * 100) + '%';
    progress.innerHTML = '正在匹配数据 ' + percent;
}

function fileSelect(e) {
    points = [];
    file = this.files[0];
    if (!file) {
        alert('文件不能为空！');
        return false;
    }
    fileLoaded = 0;
    fileSize = file.size;
    // 开始读取
    readBlob();
}
var reader = new FileReader();
// 只需监听onload事件
reader.onload = drawPoi;
input6.onchange = fileSelect;

//补足4位数  :1→0001
function padNumber(num, fill)
{
    var tmp = '0000000000' + num;
    return tmp.substring(tmp.length - fill);
}

function find_loc(lon, lat)
{
    var unit_lon = 0.00105630392291;
    var unit_lat = 0.00089831528412;
    var base_lon = 116.384251053;
    var base_lat = 30.764586;

    if (lon > 0 && lat > 0) {
        var lon_n = Math.ceil((lon - base_lon) / unit_lon);
        var lat_n = Math.ceil((lat - base_lat) / unit_lat);

        if ((0 < lon_n < 5268) && (0 < lat_n < 4858))
        {
            gid = padNumber(lon_n, 4) + padNumber(lat_n, 4);//拼接网格的id
            return gid;
        } else {
            return '';
        }
    } else {
        return '';
    }

}
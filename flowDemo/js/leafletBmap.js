var XwsoftFlow = {};

XwsoftFlow.MapControl = {
    map: null,
    gridGroup: null,
    predictData: null,
    heatmapGroup: null,
    heatmapLayer: null,
    lightRectangle: null,

    crs :  new L.Proj.CRS('EPSG:3395','+proj=merc +lon_0=0 +k=1 +x_0=140 +y_0=-250 +datum=WGS84 +units=m +no_defs',
        {
            resolutions: function () {
                level = 19
                var res = [];
                res[0] = Math.pow(2, 18);
                for (var i = 1; i < level; i++) {
                    res[i] = Math.pow(2, (18 - i))
                }
                return res;
            }(),
            origin: [0, 0],
            bounds: L.bounds([20037508.342789244, 0], [0, 20037508.342789244])
        }),

    Initial: function() {
        $('#main').height($('body').height()-55);
        this.map = L.map('main', {
            crs: this.crs,
            center: [32.047842, 118.790099],
            zoom: 13,
            layers: [
                new L.TileLayer('http://online{s}.map.bdimg.com/tile/?qt=tile&x={x}&y={y}&z={z}&styles=pl&udt=20150518', {
                    maxZoom: 18,
                    minZoom: 3,
                    subdomains: [0,1,2],
                    attribution: '',
                    tms: true
                })
            ]
        });
        this.gridGroup = L.layerGroup().addTo(this.map);
        this.heatmapGroup = L.layerGroup().addTo(this.map);
        this.AddLegendToMap();
    },
    AddLegendToMap: function() {
        var legend = L.control({ position: 'bottomleft' });

        legend.onAdd = function (map) {
            var div = L.DomUtil.create('div', 'mapInfo mapLegend'),
                grades = [10, 25, 50, 100, 150, 200, 300, 500, 700, 1000],
                labels = [],
                from, to;
            for (var i = 0; i < grades.length; i++) {
                from = grades[i];
                to = grades[i + 1];
                labels.push(
                    '<i style="background:' + XwsoftFlow.Default.GetColor(from + 1) + '"></i> ' +
                    from + (to ? '&ndash;' + to : '+'));
            }
            labels.push('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;（人）');

            div.innerHTML = labels.join('<br>');
            return div;
        };
        legend.addTo(XwsoftFlow.MapControl.map);
    },
    ShowGridLayer: function(data) {
        if (data == null) {
            alert("Data Error");
            return;
        }
        this.gridGroup.clearLayers();

        var type = $('#inOutFlowDropDown').attr('data-value');

        //网格最左下角坐标
        var point = new BMap.Point(118.62044824866, 31.926176328147);
        //横向网格数量
        var hnum = 101;
        //纵向网格数量
        var vnum = 71;

        var xLng1 = point.lng;
        var yLat1 = point.lat;
        var xLng2 = point.lng;
        var yLat2 = point.lat;
        for(var i=0; i<hnum; i++) {
            xLng1 = xLng2;
            xLng2 += 0.00105630392291 * 4;
            for(var j=0; j<vnum; j++) {
                yLat1 = yLat2;
                yLat2 += 0.00089831528412 * 4;
                var gridData = data[vnum * i + j];
                var bounds = [[yLat1, xLng1], [yLat2,xLng2]];
                if(type == 'All') {
                    gridData = gridData / 5;
                }
                var color = XwsoftFlow.Default.GetColor(gridData);
                var opacity = XwsoftFlow.Default.GetOpacity(gridData);
                var rectangle = L.rectangle(bounds, {
                    opacity: 0.5,
                    weight: 0.1,
                    color: color,
                    fillOpacity: opacity,
                    stroke: true,
                    className: ''
                });
                rectangle.on('click', function (e) {
                    // $('#PredictFlowModal').modal();
                    // $(".modal-backdrop").remove();
                    if($('#right-menu').attr('data-status') == 'closed') {
                        $('.toggler').click();
                    }
                    if(XwsoftFlow.MapControl.lightRectangle) {
                        XwsoftFlow.MapControl.map.removeLayer(XwsoftFlow.MapControl.lightRectangle)
                    }
                    XwsoftFlow.MapControl.lightRectangle = L.rectangle(this.getBounds(), {color: "#2f75e1", weight: 4, fillOpacity: 0});
                    XwsoftFlow.MapControl.lightRectangle.addTo(XwsoftFlow.MapControl.map);
                    var gridId = XwsoftFlow.Default.GetGridIdByLatLng(e.latlng.lat, e.latlng.lng);
                    XwsoftFlow.Inter.GetPredictFlowByGridId(gridId);
                });
                this.gridGroup.addLayer(rectangle);
            }
            yLat2 = point.lat;
        }
    },
    ShowHeatMap: function(data) {
        if (data == null) {
            alert("Data Error");
            return;
        }
        if(XwsoftFlow.MapControl.lightRectangle) {
            XwsoftFlow.MapControl.map.removeLayer(XwsoftFlow.MapControl.lightRectangle)
        }
        if($('#right-menu').attr('data-status') == 'opened') {
            $('.toggler').click();
        }
        var type = $('#inOutFlowDropDown').attr('data-value');
        var maxData = 1000;
        if(type == 'All') {
            maxData = 5000;
        }
        var heatmapData = {max: maxData, data:[]};

        if(this.heatmapLayer) {
            this.heatmapGroup.clearLayers();
        } else {
            var cfg = {"radius": 40,"maxOpacity": .8,"useLocalExtrema": false,latField: 'lat',lngField: 'lng',valueField: 'count',
                gradient: {
                    '.25': 'rgb(0,0,255)',
                    '.55': 'rgb(0,255,0)',
                    '.85': 'yellow',
                    '1': 'rgb(255,0,0)'
                }};
            this.heatmapLayer = new HeatmapOverlay(cfg);
        }

        //网格最左下角坐标
        var point = new BMap.Point(118.62044824866, 31.926176328147);
        //横向网格数量
        var hnum = 101;
        //纵向网格数量
        var vnum = 71;

        var xLng1 = point.lng;
        var yLat1 = point.lat;
        var xLng2 = point.lng;
        var yLat2 = point.lat;
        for(var i=0; i<hnum; i++) {
            xLng1 = xLng2;
            xLng2 += 0.00105630392291 * 4;
            for(var j=0; j<vnum; j++) {
                yLat1 = yLat2;
                yLat2 += 0.00089831528412 * 4;
                if(data[vnum * i + j] != 0) {
                    heatmapData.data.push({lat: ((yLat1 + yLat2)/2).toFixed(5)*1, lng: ((xLng1 + xLng2)/2).toFixed(5)*1, count: data[vnum * i + j]*1});
                }
            }
            yLat2 = point.lat;
        }
        this.heatmapLayer.setData(heatmapData);
        this.heatmapGroup.addLayer(this.heatmapLayer);
    },
    HideGrid: function () {
        if (this.gridGroup == null) {
            return;
        }
        this.gridGroup.clearLayers();
        $('.mapInfo.mapLegend.leaflet-control').hide();
        $('#legendSwitch').hide();
    },
    ShowGrid: function () {
        if (this.gridGroup == null) {
            return;
        }
        $('.mapInfo.mapLegend.leaflet-control').show();
        $('#legendSwitch').show();
        if(XwsoftFlow.Default.currentData) {
            XwsoftFlow.MapControl.ShowGridLayer(XwsoftFlow.Default.currentData);
        }
    },
    HideHeat: function () {
        if (this.heatmapGroup == null) {
            return;
        }
        this.heatmapGroup.clearLayers();
    },
    ShowHeat: function () {
        if (this.heatmapGroup == null) {
            return;
        }
        if(XwsoftFlow.Default.currentData) {
            XwsoftFlow.MapControl.ShowHeatMap(XwsoftFlow.Default.currentData);
        }
    }
};

XwsoftFlow.Chart = {
    chart: null,
    option: null,
    Initial: function () {
        $('#PredictFlowModal').draggable({
            handle: ".modal-header"
        });
        $('#right-menu').BootSideMenu({side:"right"});
        this.option = {
            title: {
                top: 10,
                text: '',
                left: 'center',
                textStyle: {
                    color: '#6a6a6a'
                }
            },
            grid: {
                top: 70,
                left: 50,
                right: 20
            },
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            tooltip : {
                trigger: 'axis',
                axisPointer: {
                    type: 'cross',
                    label: {
                        backgroundColor: '#6a7985'
                    }
                }
            },
            legend: {
                top: 40,
                data:['预测', '实际']
            },
            xAxis : [
                {
                    type : 'category',
                    boundaryGap : false,
                    axisLabel : {
                        interval: 1,
                        rotate: 45
                    },
                    data : []
                }
            ],
            yAxis : [
                {
                    name: '（人）       ',
                    splitLine: {
                        show:false
                    },
                    type : 'value'
                }
            ],
            series: [
            //     {
            //     name: '今日',
            //     type:'line',
            //     data: []
            // },
                {
                    itemStyle: {
                        normal: {
                            color : '#35baf5'
                        }
                    },
                    name: '预测',
                    type:'line',
                    data: []
                },{
                itemStyle: {
                    normal: {
                        color : '#ef9f83'
                    }
                },
                name: '实际',
                type:'line',
                data: []
            }]
        };
    },
    ShowChart: function(data, gridId, type) {
        if(type == 'in') {
            this.chart = echarts.init(document.getElementById('inflowDetailsCharts'));
            // this.option.backgroundColor = 'rgba(255,255,255,0.9)';
            this.option.title.text = '流入人数';
        } else if(type == 'out') {
            this.chart = echarts.init(document.getElementById('outflowDetailsCharts'));
            // this.option.backgroundColor = 'rgba(222,222,222,0.5)';
            this.option.title.text = '流出人数';
        } else {
            this.chart = echarts.init(document.getElementById('allflowDetailsCharts'));
            // this.option.backgroundColor = 'rgba(255,255,255,0.9)';
            this.option.title.text = '总人数';
        }
        for(var i=0; i<this.option.series.length; i++) {
            this.option.series[i].data = [];
        }
        document.getElementById('title').innerHTML = 'Flow 网格ID: ' + gridId;
        this.option.xAxis[0].data = data.xCategories;
        // for (var i = 0; i < data.thisdayFlow.length; i++) {
        //     this.option.series[0].data.push(data.thisdayFlow[i]*1);
        // }
        for (var i = 0; i < data.thisdayFlow.length; i++) {
            this.option.series[0].data.push(null);
        }
        for (var i = 0; i < data.predictFlow.length; i++) {
            this.option.series[0].data.push(data.predictFlow[i]*1);
        }

        for (var i = 0; i < data.yesterdayFlow.length; i++) {
            this.option.series[1].data.push(data.yesterdayFlow[i]*1);
        }
        this.chart.resize({width:$('#flowDetailsCharts').width()});
        // 使用刚指定的配置项和数据显示图表。
        this.chart.setOption(this.option, true);
    }
};

XwsoftFlow.DashBoard = {
    Initial: function () {
        $('.dashBoardItem>div').click(function (e) {
            XwsoftFlow.DashBoard.BindDashboardClick($(this).parent(), e);
        });
    },
    BindDashboardClick: function (ele, e) {
        if ($(ele).hasClass('active')) {
            return;
        }
        var items = $('.dashBoardItem');
        for (var i = 0; i < items.length; i++) {
            var src = $(items[i]).find("img").attr("src") + "";
            src = src.substr(0, src.length - "2.png".length);
            if ($(ele).attr('id') == $(items[i]).attr('id')) {
                $(items[i]).find('img').attr("src", src + "2.png");
            } else {
                $(items[i]).find('img').attr("src", src + "1.png");
            }
        }
        $(ele).siblings().removeClass('active');
        $(ele).addClass('active');
        switch ($(ele).attr('id')) {
            case 'dashBoardHand':
                XwsoftFlow.PredictTimePlayer.HidePredictPlayer();
                XwsoftFlow.Inter.GetCurrentFlow();
                break;
            case 'dashBoardPredict':
                XwsoftFlow.Inter.GetPredictFlow();
                break;
        }
    }
};

XwsoftFlow.Inter = {
    default_url : null,
    url: {
        inflow : {
            current: 'http://192.168.9.53:5000/inflow_current',
            history: 'http://192.168.9.53:5000/inflow_history',
            grid: 'http://192.168.9.53:5000/inflow_grid'
        },
        outflow : {
            current: 'http://192.168.9.53:5000/outflow_current',
            history: 'http://192.168.9.53:5000/outflow_history',
            grid: 'http://192.168.9.53:5000/outflow_grid'
        },
        allflow : {
            current: 'http://192.168.9.53:5000/allflow_current',
            history: 'http://192.168.9.53:5000/allflow_history',
            grid: 'http://192.168.9.53:5000/allflow_grid'
        }
    },
    Initial: function () {
        this.default_url = this.url.inflow;
    },
    GetCurrentFlow: function() {
        $.get(this.default_url.current, function (data) {
            data = $.parseJSON(data);
            if (data.status == "success") {
                XwsoftFlow.Default.ShowPredictTime(data.data.predictTime);
                XwsoftFlow.Default.currentData =data.data.record;
                var switchBtn=$("[name='switch']").filter(":checked").val();
                if(switchBtn == 'grid') {
                    XwsoftFlow.MapControl.ShowGridLayer(data.data.record);
                } else {
                    XwsoftFlow.MapControl.ShowHeatMap(data.data.record);
                }
            } else {
                alert(data.message);
            }
        }).fail(function (e) {
        }).always(function (e) {
        });
    },
    GetPredictFlow: function() {
        $.get(this.default_url.history, function (data) {
            data = $.parseJSON(data);
            if (data.status == "success") {
                data = data.data;
                var timeListIndex = Array.apply(null, Array(data.timeList.length)).map(function (_, i) { return i; });
                var timeListTag = data.timeList;
                XwsoftFlow.MapControl.predictData = data.valueList;

                XwsoftFlow.Default.ShowPredictTime(data.predictTime);
                XwsoftFlow.PredictTimePlayer.ShowPredictPlayer();
                XwsoftFlow.PredictTimePlayer.SetPredictPlayer(timeListIndex, timeListTag);
                XwsoftFlow.Default.currentData = data.valueList[0];
                var switchBtn=$("[name='switch']").filter(":checked").val();
                if(switchBtn == 'grid') {
                    XwsoftFlow.MapControl.ShowGridLayer(data.valueList[0]);
                } else {
                    XwsoftFlow.MapControl.ShowHeatMap(data.valueList[0]);
                }
            } else {
                alert(data.message);
            }
        }).fail(function (e) {
        }).always(function (e) {
        });
    },
    GetPredictFlowByGridId: function (gridId) {
        // $.loading('add', 'GetPredictFlowByGridId');
        $.get(this.url.inflow.grid, {grid_id:gridId}, function (data) {
            data = $.parseJSON(data);
            if (data.status == "success") {
                data = data.data;
                XwsoftFlow.Chart.ShowChart(data, gridId, 'in');
            } else {
                alert(data.message);
            }
        }).fail(function (e) {
            alert(e.message);
        }).always(function (e) {
            // $.loading('remove', 'GetPredictFlowByGridId');
        });
        $.get(this.url.outflow.grid, {grid_id:gridId}, function (data) {
            data = $.parseJSON(data);
            if (data.status == "success") {
                data = data.data;
                XwsoftFlow.Chart.ShowChart(data, gridId, 'out');
            } else {
                alert(data.message);
            }
        }).fail(function (e) {
            alert(e.message);
        }).always(function (e) {
            // $.loading('remove', 'GetPredictFlowByGridId');
        });
        $.get(this.url.allflow.grid, {grid_id:gridId}, function (data) {
            data = $.parseJSON(data);
            if (data.status == "success") {
                data = data.data;
                XwsoftFlow.Chart.ShowChart(data, gridId, 'all');
            } else {
                alert(data.message);
            }
        }).fail(function (e) {
            alert(e.message);
        }).always(function (e) {
            // $.loading('remove', 'GetPredictFlowByGridId');
        });
    }
};

XwsoftFlow.Default = {
    currentData: null,
    Initial: function () {
        $("[name='switch']").click(function(){
            var switchBtn=$("[name='switch']").filter(":checked").val();
            if(switchBtn == 'grid') {
                XwsoftFlow.MapControl.ShowGrid();
                XwsoftFlow.MapControl.HideHeat();
            } else {
                XwsoftFlow.MapControl.HideGrid();
                XwsoftFlow.MapControl.ShowHeat();
            }
        });
        XwsoftFlow.Inter.GetCurrentFlow();
        $('#legendSwitch').click(function (e) {
            if ($(this).hasClass('legendSwitchClose')) {
                $(this).removeClass('legendSwitchClose').addClass('legendSwitchOpen');
                $('.mapLegend').hide();
            } else {
                $(this).removeClass('legendSwitchOpen').addClass('legendSwitchClose');
                $('.mapLegend').show();
            }
        });
    },
    GetColor : function(d) {
        return d > 1000 ? '#a50026' :
            d > 700 ? '#d73027' :
                d > 500 ? '#f46d43' :
                    d > 300 ? '#fdae61' :
                        d > 200 ? '#fee08b' :
                            d > 150 ? '#ffffbf' :
                                d > 100 ? '#d9ef8b' :
                                    d > 50 ? '#a6d96a' :
                                        d > 25 ? '#66bd63' :
                                            d > 10 ? '#1a9850' :
                                                '#646464';
    },
    GetOpacity : function(d) {
        return d <= 10 ? 0.1 : 0.7;
    },
    GetGridIdByLatLng: function (lat, lng) {
        var grid_lng = Math.floor((lng - 118.62044824866) / 4 / 0.00105630392291 );
        var grid_lat = Math.floor((lat - 31.926176328147) / 4 / 0.00089831528412);
        var grid_lng =  ("000" + grid_lng).slice(-3);
        var grid_lat =  ("000" + grid_lat).slice(-3);
        return grid_lng + "_" + grid_lat;
    },
    ShowPredictTime: function (str) {
        $('#clock').text(moment(str, "YYYYMMDDHHmm").format("YYYY-MM-DD HH:mm"));
    }
};

XwsoftFlow.DropDown = {
    Initial: function () {
        $('.dropdownMyMenu li').click(function (e) {
            if ($(this).parents('.dropDownItem').attr('data-value') == $(this).attr('data-value')) {
                return;
            }
            $(this).parents('.dropDownItem').attr('data-value', $(this).attr('data-value'));
            $(this).parents('.dropDownItem').find('.dropdownDisplay').text($(this).text());
            XwsoftFlow.DropDown.ChangeAction();
        });
    },
    ChangeAction: function () {
        var type = $('#inOutFlowDropDown').attr('data-value');
        var div = $('.mapInfo.mapLegend.leaflet-control'),
            grades, labels = [],
            from, to, fromColor;
        if(type == 'In') {
            XwsoftFlow.Inter.default_url = XwsoftFlow.Inter.url.inflow;
            grades = [10, 25, 50, 100, 150, 200, 300, 500, 700, 1000];
        } else if(type == 'Out') {
            XwsoftFlow.Inter.default_url = XwsoftFlow.Inter.url.outflow;
            grades = [10, 25, 50, 100, 150, 200, 300, 500, 700, 1000];
        } else if(type == 'All') {
            XwsoftFlow.Inter.default_url = XwsoftFlow.Inter.url.allflow;
            grades = [50, 125, 250, 500, 750, 1000, 1500, 2500, 3500, 5000];
        }
        for (var i = 0; i < grades.length; i++) {
            from = grades[i];
            if(type == 'All') {
                fromColor = from / 5;
            } else {
                fromColor = from;
            }
            to = grades[i + 1];
            labels.push(
                '<i style="background:' + XwsoftFlow.Default.GetColor(fromColor + 1) + '"></i> ' +
                from + (to ? '&ndash;' + to : '+'));
        }
        labels.push('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;（人）');
        div.html(labels.join('<br>'));
        $('#dashBoardHand>div').click();
        XwsoftFlow.Inter.GetCurrentFlow();
    }
};

XwsoftFlow.PredictTimePlayer = {
    IsSetPreditTimePlayer: false,

    Initial: function () {
        $('#PredictPlayer .timePlayerBtn').click(function (e) {
            if ($(this).hasClass('Play')) {
                $(this).removeClass('Play').attr('src', 'img/pause.png');
                XwsoftFlow.PredictTimePlayer.PlayPredictFlow();
            } else {
                $(this).addClass('Play').attr('src', 'img/play.png');
            }
        });
        $(window).resize(function (e) {
            XwsoftFlow.PredictTimePlayer.SetTimeLineWidth();
        });
    },

    PlayPredictFlow: function () {
        if ($('#PredictPlayer .timePlayerBtn').hasClass('Play') || !$('#PredictPlayer').is(':visible')) {
            return;
        }
        var currentValue = $('#PredictPlayer input').slider('getValue');
        var allValueArr = $('#PredictPlayer input').slider('getAttribute', 'ticks');
        var nextValue = (currentValue + 1) % allValueArr.length;
        $('#PredictPlayer input').slider('setValue', nextValue, false, true);
        setTimeout(function () {
            if ($('#PredictPlayer .timePlayerBtn').hasClass('Play') || !$('#PredictPlayer').is(':visible')) {
                $('#PredictPlayer .timePlayerBtn').addClass('Play').attr('src', 'img/play.png');
                return;
            }
            XwsoftFlow.PredictTimePlayer.PlayPredictFlow();
        }, 1000);
    },

    SetTimeLineWidth: function () {
        if (XwsoftFlow.PredictTimePlayer.IsSetPreditTimePlayer)
        {
            $('.timePlayer .slider.slider-horizontal').width($('.timePlayer').width() - 108);
            $('.timePlayer input').slider('relayout');
        }
    },
    SetPredictPlayer: function(valueArr, labelArr) {
        if (labelArr.length != valueArr.length) {
            return;
        }
        var ticks = valueArr;
        var labels = labelArr;
        if (XwsoftFlow.PredictTimePlayer.IsSetPreditTimePlayer) {
            $('#PredictPlayer input').slider('destroy');
        }
        $('#PredictPlayer input').slider({
            ticks: ticks,
            ticks_labels: labels,
            step: 1,
            value: 0,
            tooltip: "hide"
        }).slider('on', 'change', function (e) {
            XwsoftFlow.Default.currentData = XwsoftFlow.MapControl.predictData[e.newValue];
            var switchBtn=$("[name='switch']").filter(":checked").val();
            if(switchBtn == 'grid') {
                XwsoftFlow.MapControl.ShowGridLayer(XwsoftFlow.MapControl.predictData[e.newValue]);
            } else {
                XwsoftFlow.MapControl.ShowHeatMap(XwsoftFlow.MapControl.predictData[e.newValue]);
            }
        });
        XwsoftFlow.PredictTimePlayer.IsSetPreditTimePlayer = true;
        this.SetTimeLineWidth();
    },
    ShowPredictPlayer: function() {
        $('#PredictPlayer').show();
    },
    HidePredictPlayer: function() {
        $('#PredictPlayer').hide();
    }
};

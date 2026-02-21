(function () {
  "use strict";

  angular.module("inPhaseApp").controller("NdtController", [
    "ApiService",
    function (ApiService) {
      var vm = this;

      vm.loading = true;
      vm.detailLoading = false;
      vm.signalLoading = false;
      vm.error = null;
      vm.signalError = null;
      vm.samples = [];
      vm.searchQuery = "";
      vm.selectedName = null;
      vm.selected = null;
      vm.signal = null;
      vm.maxSignalPoints = 1024;
      vm.chart = null;

      vm.filteredSamples = function () {
        if (!vm.searchQuery) {
          return vm.samples;
        }
        var query = vm.searchQuery.toLowerCase();
        return vm.samples.filter(function (sample) {
          return sample.name.toLowerCase().indexOf(query) !== -1;
        });
      };

      vm.refreshSignal = function () {
        if (!vm.selectedName) {
          return;
        }
        loadSignal(vm.selectedName);
      };

      function buildSignalChart(signal) {
        var width = 920;
        var height = 260;
        var padX = 16;
        var padY = 16;
        var xMin = signal.time_us[0];
        var xMax = signal.time_us[signal.time_us.length - 1];
        var yMin = signal.stats.amplitude_min;
        var yMax = signal.stats.amplitude_max;

        if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) {
          yMin = -1;
          yMax = 1;
        }
        if (Math.abs(yMax - yMin) < 1e-12) {
          yMin -= 1.0;
          yMax += 1.0;
        }
        if (Math.abs(xMax - xMin) < 1e-12) {
          xMin = 0.0;
          xMax = 1.0;
        }

        var plotWidth = width - padX * 2;
        var plotHeight = height - padY * 2;

        function toX(t) {
          return padX + ((t - xMin) / (xMax - xMin)) * plotWidth;
        }

        function toY(v) {
          return padY + (1.0 - (v - yMin) / (yMax - yMin)) * plotHeight;
        }

        var path = "";
        for (var i = 0; i < signal.time_us.length; i += 1) {
          var x = toX(signal.time_us[i]);
          var y = toY(signal.rf[i]);
          if (i === 0) {
            path += "M " + x.toFixed(2) + " " + y.toFixed(2);
          } else {
            path += " L " + x.toFixed(2) + " " + y.toFixed(2);
          }
        }

        var markers = (signal.defect_markers || []).map(function (marker) {
          var ampStr = marker.amplitude != null ? " | A=" + marker.amplitude.toFixed(3) : "";
          var confidenceStr = marker.confidence != null ? " | conf=" + marker.confidence.toFixed(2) : "";
          var sourceStr = marker.source ? " | " + marker.source : "";
          return {
            x: toX(marker.two_way_time_us),
            label:
              marker.depth_mm.toFixed(2) +
              " mm | t=" +
              marker.two_way_time_us.toFixed(3) +
              " µs" +
              ampStr +
              confidenceStr +
              sourceStr,
          };
        });

        var wallMarkers = (signal.wall_markers || []).map(function (marker) {
          var depthStr = marker.depth_mm != null ? marker.depth_mm.toFixed(2) + " mm | " : "";
          var ampStr = marker.amplitude != null ? " | A=" + marker.amplitude.toFixed(3) : "";
          return {
            x: toX(marker.two_way_time_us),
            label:
              marker.label.replace("_", " ") +
              " | " +
              depthStr +
              "t=" +
              marker.two_way_time_us.toFixed(3) +
              " \u00b5s" +
              ampStr,
          };
        });

        return {
          width: width,
          height: height,
          path: path,
          baselineY: toY(0.0),
          markers: markers,
          wallMarkers: wallMarkers,
        };
      }

      function loadSignal(sampleName) {
        vm.signalLoading = true;
        vm.signalError = null;
        ApiService.getNdtSignal(sampleName, vm.maxSignalPoints)
          .then(function (data) {
            vm.signal = data;
            vm.chart = buildSignalChart(data);
          })
          .catch(function (error) {
            vm.signalError = error.detail || "Failed to load NDT waveform preview";
            vm.signal = null;
            vm.chart = null;
          })
          .finally(function () {
            vm.signalLoading = false;
          });
      }

      vm.loadDetail = function (sampleName) {
        vm.selectedName = sampleName;
        vm.detailLoading = true;
        vm.error = null;
        vm.signalError = null;
        vm.signal = null;
        vm.chart = null;

        ApiService.getNdtSample(sampleName)
          .then(function (detail) {
            vm.selected = detail;
            loadSignal(sampleName);
          })
          .catch(function (error) {
            vm.error = error.detail || "Failed to load sample details";
            vm.selected = null;
          })
          .finally(function () {
            vm.detailLoading = false;
          });
      };

      function init() {
        vm.loading = true;
        vm.error = null;

        ApiService.listNdtSamples()
          .then(function (data) {
            vm.samples = data;
            if (vm.samples.length > 0) {
              vm.loadDetail(vm.samples[0].name);
            }
          })
          .catch(function (error) {
            vm.error = error.detail || "Failed to load NDT sample list";
          })
          .finally(function () {
            vm.loading = false;
          });
      }

      init();
    },
  ]);
})();

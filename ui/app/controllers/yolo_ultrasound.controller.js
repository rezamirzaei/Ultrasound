(function () {
  "use strict";

  angular.module("inPhaseApp").controller("YoloUltrasoundController", [
    "ApiService",
    "$q",
    function (ApiService, $q) {
      var vm = this;

      vm.loading = true;
      vm.error = null;
      vm.status = null;
      vm.canDownload = ApiService.hasRole("analyst");

      vm.downloadLoading = false;
      vm.downloadError = null;

      vm.sampleLoading = false;
      vm.sampleError = null;
      vm.sample = null;

      vm.predictionLoading = false;
      vm.predictionError = null;
      vm.prediction = null;

      vm.sampleForm = {
        class_name: "benign",
        sample_index: 0,
      };

      vm.predictForm = {
        model: "",
        confidence: 0.25,
        iou_threshold: 0.45,
        image_size: 640,
        max_detections: 100,
      };

      vm.refreshStatus = refreshStatus;
      vm.downloadModel = downloadModel;
      vm.loadSample = loadSample;
      vm.runPrediction = runPrediction;

      function refreshStatus() {
        return ApiService.getBusiYoloStatus()
          .then(function (payload) {
            vm.status = payload;

            var recommended = payload && payload.model && payload.model.local_path ? payload.model.local_path : "";
            var downloaded = payload && payload.model && payload.model.downloaded;
            var defaults = (payload && payload.yolo && payload.yolo.default_models) || [];

            var currentModel = vm.predictForm.model || "";
            if (downloaded && recommended) {
              if (!currentModel || (defaults && defaults.indexOf(currentModel) !== -1) || currentModel === "yolov8n.pt") {
                vm.predictForm.model = recommended;
              }
              return;
            }

            if (!currentModel) {
              if (defaults && defaults.length) {
                vm.predictForm.model = defaults[0];
              } else {
                vm.predictForm.model = "yolov8n.pt";
              }
            }
          })
          .catch(function (error) {
            vm.status = null;
            vm.error = error.detail || "Failed to load ultrasound YOLO lab status";
          });
      }

      function downloadModel(force) {
        if (!vm.canDownload) {
          return;
        }
        vm.downloadLoading = true;
        vm.downloadError = null;
        return ApiService.downloadBusiYoloModel(!!force)
          .then(function () {
            return refreshStatus();
          })
          .catch(function (error) {
            vm.downloadError = error.detail || "Model download failed";
          })
          .finally(function () {
            vm.downloadLoading = false;
          });
      }

      function loadSample() {
        vm.sampleLoading = true;
        vm.sampleError = null;
        vm.sample = null;
        vm.prediction = null;
        vm.predictionError = null;

        var className = vm.sampleForm.class_name || "benign";
        var sampleIndex = Math.max(0, Math.floor(vm.sampleForm.sample_index || 0));

        return ApiService.getBusiYoloSample(className, sampleIndex)
          .then(function (payload) {
            vm.sample = payload;
          })
          .catch(function (error) {
            vm.sampleError = error.detail || "Failed to load BUSI sample";
          })
          .finally(function () {
            vm.sampleLoading = false;
          });
      }

      function runPrediction() {
        if (!vm.sample || !vm.sample.sample) {
          return;
        }
        vm.predictionLoading = true;
        vm.predictionError = null;
        vm.prediction = null;

        var className = vm.sample.sample.class_name;
        var sampleIndex = vm.sample.sample.requested_index;

        var payload = {
          model: vm.predictForm.model,
          confidence: Number(vm.predictForm.confidence || 0.25),
          iou_threshold: Number(vm.predictForm.iou_threshold || 0.45),
          image_size: Math.max(160, Math.floor(vm.predictForm.image_size || 640)),
          max_detections: Math.max(1, Math.floor(vm.predictForm.max_detections || 100)),
        };

        return ApiService.predictBusiYoloSample(className, sampleIndex, payload)
          .then(function (prediction) {
            vm.prediction = prediction;
          })
          .catch(function (error) {
            vm.predictionError = error.detail || "Prediction failed";
          })
          .finally(function () {
            vm.predictionLoading = false;
          });
      }

      function init() {
        vm.loading = true;
        vm.error = null;
        $q.all([refreshStatus()])
          .then(function () {
            return loadSample();
          })
          .finally(function () {
            vm.loading = false;
          });
      }

      init();
    },
  ]);
})();

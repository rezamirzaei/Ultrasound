(function () {
  "use strict";

  angular.module("inPhaseApp").controller("YoloController", [
    "ApiService",
    "$q",
    function (ApiService, $q) {
      var vm = this;

      vm.loading = true;
      vm.error = null;
      vm.status = null;

      vm.sampleLoading = false;
      vm.sampleError = null;
      vm.sample = null;

      vm.predictionLoading = false;
      vm.predictionError = null;
      vm.prediction = null;

      vm.sampleForm = {
        category: "Benign",
        sample_index: 0,
      };

      vm.predictForm = {
        model: "yolo11n.pt",
        confidence: 0.25,
        iou_threshold: 0.45,
        image_size: 640,
        max_detections: 100,
      };

      vm.refreshStatus = refreshStatus;
      vm.loadSample = loadSample;
      vm.runPrediction = runPrediction;

      function refreshStatus() {
        return ApiService.getLiverYoloStatus()
          .then(function (payload) {
            vm.status = payload;
          })
          .catch(function (error) {
            vm.status = null;
            vm.error = error.detail || "Failed to load liver YOLO lab status";
          });
      }

      function loadSample() {
        vm.sampleLoading = true;
        vm.sampleError = null;
        vm.sample = null;
        vm.prediction = null;
        vm.predictionError = null;

        var category = vm.sampleForm.category || "Benign";
        var index = Math.max(0, Math.floor(vm.sampleForm.sample_index || 0));

        return ApiService.getLiverSample(category, index)
          .then(function (payload) {
            vm.sample = payload;
          })
          .catch(function (error) {
            vm.sampleError = error.detail || "Failed to load liver sample";
          })
          .finally(function () {
            vm.sampleLoading = false;
          });
      }

      function runPrediction() {
        if (!vm.sample) {
          return;
        }
        vm.predictionLoading = true;
        vm.predictionError = null;
        vm.prediction = null;

        var category = vm.sample.category;
        var index = vm.sample.sample_index;

        var payload = {
          model: vm.predictForm.model,
          confidence: Number(vm.predictForm.confidence || 0.25),
          iou_threshold: Number(vm.predictForm.iou_threshold || 0.45),
          image_size: Math.max(160, Math.floor(vm.predictForm.image_size || 640)),
          max_detections: Math.max(1, Math.floor(vm.predictForm.max_detections || 100)),
        };

        return ApiService.predictLiverSample(category, index, payload)
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

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
      vm.canUpload = ApiService.hasRole("analyst");

      vm.records = [];
      vm.recordsLoading = false;
      vm.recordsError = null;

      vm.record = null;
      vm.recordLoading = false;
      vm.recordError = null;

      vm.uploadLoading = false;
      vm.uploadError = null;
      vm.uploadResult = null;

      vm.predictionLoading = false;
      vm.predictionError = null;
      vm.prediction = null;

      vm.uploadForm = {
        asset_id: "pipe-rack-001",
        location_name: "Unit A",
        latitude: null,
        longitude: null,
        sensor: "camera",
        class_names: "anomaly",
        inspector: "",
        notes: "",
      };

      vm.predictForm = {
        model: "yolov8n.pt",
        confidence: 0.25,
        iou_threshold: 0.45,
        image_size: 640,
        max_detections: 100,
      };

      vm.refreshRecords = refreshRecords;
      vm.openRecord = openRecord;
      vm.uploadRecord = uploadRecord;
      vm.runPrediction = runPrediction;

      function toClassNames(raw) {
        if (!raw) {
          return ["anomaly"];
        }
        var parts = String(raw)
          .split(",")
          .map(function (item) {
            return item.trim();
          })
          .filter(function (item) {
            return !!item;
          });
        return parts.length ? parts : ["anomaly"];
      }

      function refreshStatus() {
        return ApiService.getYoloStatus()
          .then(function (payload) {
            vm.status = payload;
            if (payload && payload.default_models && payload.default_models.length) {
              vm.predictForm.model = payload.default_models[0];
            }
          })
          .catch(function (error) {
            vm.status = null;
            vm.error = error.detail || "Failed to load YOLO backend status";
          });
      }

      function refreshRecords() {
        vm.recordsLoading = true;
        vm.recordsError = null;
        return ApiService.listFieldYoloRecords(80)
          .then(function (payload) {
            vm.records = payload || [];
            if (!vm.record && vm.records.length) {
              openRecord(vm.records[0].record_id);
            }
          })
          .catch(function (error) {
            vm.recordsError = error.detail || "Failed to load field records";
          })
          .finally(function () {
            vm.recordsLoading = false;
          });
      }

      function openRecord(recordId) {
        if (!recordId) {
          return;
        }
        vm.recordLoading = true;
        vm.recordError = null;
        vm.prediction = null;
        vm.predictionError = null;

        ApiService.getFieldYoloRecord(recordId)
          .then(function (payload) {
            vm.record = payload;
          })
          .catch(function (error) {
            vm.recordError = error.detail || "Failed to load record details";
          })
          .finally(function () {
            vm.recordLoading = false;
          });
      }

      function uploadRecord() {
        if (!vm.canUpload) {
          return;
        }
        vm.uploadLoading = true;
        vm.uploadError = null;
        vm.uploadResult = null;

        var imageInput = document.getElementById("yolo-field-upload-image");
        if (!imageInput || !imageInput.files || !imageInput.files.length) {
          vm.uploadLoading = false;
          vm.uploadError = "Select an image file before upload.";
          return;
        }

        var labelsInput = document.getElementById("yolo-field-upload-labels");
        var metadata = {
          asset_id: vm.uploadForm.asset_id,
          location_name: vm.uploadForm.location_name || null,
          latitude: vm.uploadForm.latitude == null ? null : Number(vm.uploadForm.latitude),
          longitude: vm.uploadForm.longitude == null ? null : Number(vm.uploadForm.longitude),
          captured_at: new Date().toISOString(),
          inspector: vm.uploadForm.inspector || null,
          sensor: vm.uploadForm.sensor || "camera",
          class_names: toClassNames(vm.uploadForm.class_names),
          notes: vm.uploadForm.notes || null,
          extra: {},
        };

        var formData = new FormData();
        formData.append("metadata_json", JSON.stringify(metadata));
        formData.append("image", imageInput.files[0]);
        if (labelsInput && labelsInput.files && labelsInput.files.length) {
          formData.append("labels", labelsInput.files[0]);
        }

        ApiService.uploadFieldYoloRecord(formData)
          .then(function (payload) {
            vm.uploadResult = payload;
            imageInput.value = "";
            if (labelsInput) {
              labelsInput.value = "";
            }
            return refreshRecords().then(function () {
              openRecord(payload.record_id);
            });
          })
          .catch(function (error) {
            vm.uploadError = error.detail || "Upload failed";
          })
          .finally(function () {
            vm.uploadLoading = false;
          });
      }

      function runPrediction() {
        if (!vm.record || !vm.record.record_id) {
          return;
        }
        vm.predictionLoading = true;
        vm.predictionError = null;
        vm.prediction = null;

        var payload = {
          model: vm.predictForm.model,
          confidence: Number(vm.predictForm.confidence || 0.25),
          iou_threshold: Number(vm.predictForm.iou_threshold || 0.45),
          image_size: Math.max(160, Math.floor(vm.predictForm.image_size || 640)),
          max_detections: Math.max(1, Math.floor(vm.predictForm.max_detections || 100)),
        };

        ApiService.predictFieldYoloRecord(vm.record.record_id, payload)
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
        $q.all([refreshStatus(), refreshRecords()])
          .finally(function () {
            vm.loading = false;
          });
      }

      init();
    },
  ]);
})();


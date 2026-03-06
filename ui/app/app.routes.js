(function () {
  "use strict";

  angular.module("inPhaseApp").config([
    "$routeProvider",
    "$locationProvider",
    function ($routeProvider, $locationProvider) {
      $locationProvider.hashPrefix("");

      $routeProvider
        .when("/dashboard", {
          templateUrl: "app/views/dashboard.html",
          controller: "DashboardController",
          controllerAs: "vm",
        })
        .when("/busi", {
          templateUrl: "app/views/busi.html",
          controller: "BusiController",
          controllerAs: "vm",
        })
        .when("/preprocessing", {
          templateUrl: "app/views/preprocessing.html",
          controller: "PreprocessingController",
          controllerAs: "vm",
          requiredRole: "analyst",
        })
        .when("/phase-retrieval", {
          templateUrl: "app/views/phase_retrieval.html",
          controller: "PhaseRetrievalController",
          controllerAs: "vm",
        })
        .when("/ndt", {
          templateUrl: "app/views/ndt.html",
          controller: "NdtController",
          controllerAs: "vm",
        })
        .when("/industrial", {
          templateUrl: "app/views/industrial.html",
          controller: "IndustrialController",
          controllerAs: "vm",
        })
        .when("/yolo", {
          templateUrl: "app/views/yolo.html",
          controller: "YoloController",
          controllerAs: "vm",
          requiredRole: "viewer",
        })
        .when("/yolo-ultrasound", {
          templateUrl: "app/views/yolo_ultrasound.html",
          controller: "YoloUltrasoundController",
          controllerAs: "vm",
          requiredRole: "viewer",
        })
        .otherwise({ redirectTo: "/dashboard" });
    },
  ]);
})();

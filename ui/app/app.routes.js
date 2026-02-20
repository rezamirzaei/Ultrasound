(function () {
  "use strict";

  angular.module("inPhaseApp").config([
    "$routeProvider",
    function ($routeProvider) {
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
        })
        .when("/ndt", {
          templateUrl: "app/views/ndt.html",
          controller: "NdtController",
          controllerAs: "vm",
        })
        .otherwise({ redirectTo: "/dashboard" });
    },
  ]);
})();

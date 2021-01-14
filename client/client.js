function getPortfolio(e) {
	e.preventDefault();
	$("#txtarea").text("");
	$("#txtFutureCapital").val("");
	$(".loader").css("visibility", "visible");
	var capital = $('#txtCapital').val();
	var risk = $('#txtRisk').val();
	$.ajax({
		url: "https://localhost:5001/api/indici/?capital=" + capital + "&riskAlpha=" + risk,
		type: "GET",
		contentType: "application/json",
		data: "",
		success: function(result) {
			console.log(result);
			showResult(JSON.parse(result));
		},
		error: function(xhr, status, p3, p4) {
			var err = "Error " + " " + status + " " + p3;
			if (xhr.responseText && xhr.responseText[0] == "{")
				err = JSON.parse(xhr.responseText).message;
			alert(err);
		}
	});
}

function showResult(res) {
	$(".loader").css("visibility", "hidden");
	$(".input-group").css("visibility", "visible");
	$("#txtarea").text(JSON.stringify(res.portfolio, null, 2));
	$("#txtFutureCapital").val(res.return);
}

function init() {
	
}

function findAll() {
	$.ajax( {
		url: "https://localhost:5001/api/Stagione",
		type: "GET",
		contentType: "application/json",
		success: function (result) {
			console.log(result);
			readResult(JSON.stringify(result));
		},
		error: function (xhr, status, p3, p4) {
			var err = "Error " + " " + status + " " + p3;
			if (xhr.responseText && xhr.responseText[0] == "{")
			err = JSON.parse(xhr.responseText).message;
			alert(err);
		}
	});
}

function findById() {
	var id = $('#txtId').val();
	$.ajax( {
		url: "https://localhost:5001/api/Stagione/" + id,
		type: "GET",
		contentType: "application/json",
		data: "",
		success: function (result) {
			console.log(result);
			readResult(JSON.stringify(result));
		},
		error: function (xhr, status, p3, p4) {
			var err = "Error " + " " + status + " " + p3;
			if (xhr.responseText && xhr.responseText[0] == "{")
			err = JSON.parse(xhr.responseText).message;
			alert(err);
		}
	});
}

function postItem() {
	var id = $('#txtId').val();
	var anno = $('#txtNewAnno').val();
	var options = {};
	options.url = "https://localhost:5001/api/Stagione/PostStagioneItem";
	options.type = "POST";
	options.data = JSON.stringify({
		"id": Number(id),
		"anno": Number(anno),
		"serie": 'C'
	});
	options.dataType = "json";
	options.contentType = "application/json";
	options.success = function (msg) {
		alert(msg);
	};
	options.error = function (err) {
		alert(err.responseText);
	};
	$.ajax(options);
}

function deleteId() {
	var options = {};
	options.url = "https://localhost:5001/api/Stagione/"+ $("#txtId").val();
	options.type = "DELETE";
	options.contentType = "application/json";
	options.success = function (msg) {
		alert(msg);
	};
	options.error = function (err) { alert(err.statusText); };
	$.ajax(options);
}

function updateId() {
	var id = $('#txtId').val();
	var anno = $('#txtNewAnno').val();
	var options = {};
	options.url = "https://localhost:5001/api/Stagione/"+ $("#txtId").val();
	options.type = "PUT";
	options.data = JSON.stringify({
		"id": Number(id),
		"anno": Number(anno),
		"serie": 'C'
	});
	options.dataType = "json";
	options.contentType = "application/json";
	options.success = function (msg) { alert(msg); };
	options.error = function (err) { alert(err.responseText); };
	$.ajax(options);
};

function readResult(str) { // Gestisce i dati restituiti dal server.
	document.getElementById('txtarea').value += str;
	console.log(str);
}

function getIndexById() {
	var id = $('#txtId').val();
	$.ajax({
		url: "https://localhost:5001/api/indici/" + id,
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
	document.getElementById('txtarea').value = "";
	document.getElementById('txtarea').value += res.text;
	renderImage(res.img);
}

function renderImage(base64imageString) {
	var baseStr64 = base64imageString;
	baseStr64 = baseStr64.substring(0, baseStr64.length-1); // tolgo l'ultimo carattere della stringa che codifica l'immagine ("'")
	baseStr64 = baseStr64.substring(2, baseStr64.length); // tolgo i primi due caratteri della stringa che codifica l'immagine ("b'")
	var image = new Image();
	image.src = 'data:image/png;base64,' + baseStr64; // dico che l'immagine è una png codificata in base64 e fornisco l'immagine codificata vera e propria ("baseStr64")
	document.body.appendChild(image);
}
	
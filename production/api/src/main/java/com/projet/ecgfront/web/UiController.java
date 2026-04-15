package com.projet.ecgfront.web;

import com.projet.ecgfront.service.AiGatewayService;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.client.HttpStatusCodeException;
import org.springframework.web.multipart.MultipartFile;

@Controller
public class UiController {

    private static final List<String> MODELS = List.of("mlp", "cnn", "rnn");
    private final AiGatewayService aiGatewayService;

    public UiController(AiGatewayService aiGatewayService) {
        this.aiGatewayService = aiGatewayService;
    }

    @GetMapping("/")
    public String index(Model model) {
        seed(model, "cnn");
        return "index";
    }

    @PostMapping("/classify")
    public String classify(
            @RequestParam("model") String model,
            @RequestParam("signalFile") MultipartFile signalFile,
            Model view
    ) {
        String selected = normalizeModel(model);
        seed(view, selected);

        if (signalFile == null || signalFile.isEmpty()) {
            view.addAttribute("error", "Ajoute un fichier signal avant classification.");
            return "index";
        }

        try {
            ResponseEntity<Map<String, Object>> response = aiGatewayService.classify(selected, signalFile);
            view.addAttribute("statusCode", response.getStatusCode().value());
            view.addAttribute("singleResult", response.getBody());
        } catch (HttpStatusCodeException ex) {
            view.addAttribute("statusCode", ex.getStatusCode().value());
            view.addAttribute("error", ex.getResponseBodyAsString());
        } catch (IOException ex) {
            view.addAttribute("statusCode", 400);
            view.addAttribute("error", "Lecture fichier impossible");
        } catch (Exception ex) {
            view.addAttribute("statusCode", 502);
            view.addAttribute("error", "Service IA indisponible");
        }

        return "index";
    }

    @PostMapping("/classify-all")
    public String classifyAll(
            @RequestParam("signalFile") MultipartFile signalFile,
            Model view
    ) {
        seed(view, "cnn");

        if (signalFile == null || signalFile.isEmpty()) {
            view.addAttribute("error", "Ajoute un fichier signal avant classification.");
            return "index";
        }

        try {
            ResponseEntity<Map<String, Object>> response = aiGatewayService.classifyAll(signalFile);
            view.addAttribute("statusCode", response.getStatusCode().value());
            view.addAttribute("allResult", response.getBody());
        } catch (HttpStatusCodeException ex) {
            view.addAttribute("statusCode", ex.getStatusCode().value());
            view.addAttribute("error", ex.getResponseBodyAsString());
        } catch (IOException ex) {
            view.addAttribute("statusCode", 400);
            view.addAttribute("error", "Lecture fichier impossible");
        } catch (Exception ex) {
            view.addAttribute("statusCode", 502);
            view.addAttribute("error", "Service IA indisponible");
        }

        return "index";
    }

    private void seed(Model view, String selected) {
        view.addAttribute("modelOptions", MODELS);
        view.addAttribute("selectedModel", selected);
    }

    private String normalizeModel(String incoming) {
        if (incoming == null) {
            return "cnn";
        }
        String cleaned = incoming.strip().toLowerCase();
        return MODELS.contains(cleaned) ? cleaned : "cnn";
    }
}

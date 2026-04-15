package com.projet.ecgfront.web;

import com.projet.ecgfront.service.AiGatewayService;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.client.HttpStatusCodeException;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api")
public class ApiController {

    private final AiGatewayService aiGatewayService;

    public ApiController(AiGatewayService aiGatewayService) {
        this.aiGatewayService = aiGatewayService;
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("status", "ok");
        payload.put("message", "spring-front ready");
        return payload;
    }

    @PostMapping("/classify")
    public ResponseEntity<?> classify(
            @RequestParam("model") String model,
            @RequestParam("signal") MultipartFile signal
    ) {
        if (signal == null || signal.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "Ajoutez un fichier signal"));
        }

        try {
            ResponseEntity<Map<String, Object>> response = aiGatewayService.classify(model, signal);
            return ResponseEntity.status(response.getStatusCode()).body(response.getBody());
        } catch (HttpStatusCodeException ex) {
            return ResponseEntity.status(ex.getStatusCode()).body(Map.of("error", ex.getResponseBodyAsString()));
        } catch (IOException ex) {
            return ResponseEntity.badRequest().body(Map.of("error", "Lecture fichier impossible"));
        } catch (Exception ex) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body(Map.of("error", "Service IA indisponible"));
        }
    }

    @PostMapping("/classify-all")
    public ResponseEntity<?> classifyAll(@RequestParam("signal") MultipartFile signal) {
        if (signal == null || signal.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "Ajoutez un fichier signal"));
        }

        try {
            ResponseEntity<Map<String, Object>> response = aiGatewayService.classifyAll(signal);
            return ResponseEntity.status(response.getStatusCode()).body(response.getBody());
        } catch (HttpStatusCodeException ex) {
            return ResponseEntity.status(ex.getStatusCode()).body(Map.of("error", ex.getResponseBodyAsString()));
        } catch (IOException ex) {
            return ResponseEntity.badRequest().body(Map.of("error", "Lecture fichier impossible"));
        } catch (Exception ex) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body(Map.of("error", "Service IA indisponible"));
        }
    }
}
